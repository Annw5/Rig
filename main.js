import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { FaceLandmarker, PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// ---------- Сцена, камера, рендерер ----------
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x222222);

const camera = new THREE.PerspectiveCamera(
  30,
  window.innerWidth / window.innerHeight,
  0.1,
  20
);
camera.position.set(0, 0.85, 4.2);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// ---------- Свет ----------
const light = new THREE.DirectionalLight(0xffffff, 2);
light.position.set(1, 1, 1);
scene.add(light);
scene.add(new THREE.AmbientLight(0xffffff, 0.8));

// ---------- Элементы страницы ----------
const fileInput = document.getElementById('fileInput');
const video = document.getElementById('video');
const cameraButton = document.getElementById('cameraButton');
const calibrateButton = document.getElementById('calibrateButton');
const resetButton = document.getElementById('resetButton');
const debugBox = document.getElementById('debug');
const exprBox = document.getElementById('exprBox');
const bgButton = document.getElementById('bgButton');
const armsButton = document.getElementById('bodyButton'); // та же кнопка, теперь только для рук
const panel = document.getElementById('panel');

// ---------- Фон ----------
const BG_MODES = ['dark', 'green', 'transparent'];
const BG_NAMES = { dark: 'тёмный', green: 'зелёный', transparent: 'прозрачный' };

let bgMode = localStorage.getItem('bgMode') ?? 'dark';
if (!BG_MODES.includes(bgMode)) {
  bgMode = 'dark';
}

function applyBackground() {
  if (bgMode === 'green') {
    scene.background = new THREE.Color(0x00ff00);
    document.body.style.background = '#00ff00';
  } else if (bgMode === 'transparent') {
    scene.background = null;
    document.body.style.background = 'transparent';
  } else {
    scene.background = new THREE.Color(0x222222);
    document.body.style.background = '#222';
  }
  bgButton.textContent = 'Фон: ' + BG_NAMES[bgMode];
  localStorage.setItem('bgMode', bgMode);
}
applyBackground();

bgButton.addEventListener('click', () => {
  const next = (BG_MODES.indexOf(bgMode) + 1) % BG_MODES.length;
  bgMode = BG_MODES[next];
  applyBackground();
});

window.addEventListener('keydown', (event) => {
  if (event.code === 'KeyH') {
    panel.hidden = !panel.hidden;
  }
});

// ---------- Настройки ----------
let neutral = {};
let calibration = null;

try {
  neutral = JSON.parse(localStorage.getItem('neutral')) ?? {};
} catch (error) {
  neutral = {};
}

// Нулевые точки тела. Не сохраняются между запусками
let base = {};
const BODY_KEYS = [
  'headX', 'headY', 'headZ',
  'gapLeft', 'gapRight',
  'depthLeft', 'depthRight',
  'torsoRoll', 'torsoYaw'
];

const RANGE = {
  blink: 0.5,
  pucker: 0.5,
  smile: 0.5,
  brow: 0.5
};
const DEAD = 0.08;

// Рот (автоподстройка)
const DEAD_JAW = 0.02;
const JAW_MAX_BASE = 0.03;
const JAW_MIN_PEAK = 0.08;
const JAW_DECAY = 0.99;
let jawPeak = JAW_MIN_PEAK;

// Голова (повороты)
const YAW_SIGN = -1;
const PITCH_SIGN = 1;
const ROLL_SIGN = -1;
const NECK_SHARE = 0.3;
const HEAD_SMOOTH = 0.4;
const MAX_ANGLE = { yaw: 1.0, pitch: 0.8, roll: 0.7 };
const headTarget = { yaw: 0, pitch: 0, roll: 0 };

// Голова (смещение только головы и шеи, не всего тела). Единицы: см -> метры
const HEAD_POS_SIGN = { x: -1, y: 1, z: 1 };
const HEAD_POS_GAIN = { x: 0.008, y: 0.008, z: 0.01 };
const HEAD_POS_MAX = { x: 0.15, y: 0.1, z: 0.2 };
const POS_SMOOTH = 0.15;
const headOffset = { x: 0, y: 0, z: 0 };

// Потеря лица
const MAX_VALID = { yaw: 1.4, pitch: 1.1, roll: 1.0 };
const LOST_DELAY = 400;
const RELAX_SMOOTH = 0.05;
let lastFaceTime = 0;

// Руки
const ARM_SIGN = { x: -1, y: -1, z: -1 };
const POSE_VISIBLE = 0.5;
const ARM_SMOOTH = 0.3;
const ARM_REST_OUT = 0.25;
const POSE_EVERY = 2;

// Плечи: подъём и опускание (у каждого плеча отдельно)
const SHOULDER_SIGN = 1;
const SHOULDER_GAIN = 3;
const SHOULDER_DEAD = 0.02;
const SHOULDER_MAX = 0.5;
const SHOULDER_SMOOTH = 0.3;

// Плечи: глубина (вперёд и назад, у каждого плеча отдельно)
const SHOULDER_DEPTH_SIGN = -1; // если плечо уходит не в ту сторону, поставь 1
const SHOULDER_DEPTH_GAIN = 1;
const SHOULDER_DEPTH_DEAD = 0.02;
const SHOULDER_DEPTH_MAX = 0.15;

// Туловище: наклон вбок и поворот (по плечам). Вперёд и назад не наклоняется
const TORSO_ROLL_SIGN = 1;
const TORSO_YAW_SIGN = -1;
const TORSO_DEAD = 0.03;
const TORSO_MAX = { yaw: 0.8, roll: 0.6 };
const TORSO_SMOOTH = 0.2;
const torso = { yaw: 0, roll: 0 };

// Брови
const SORROW_GAIN = 1.0;
const BROW_MORPHS = ['Fcl_BRW_Surprised', 'Fcl_BRW_Angry', 'Fcl_BRW_Sorrow'];
const BROW_SEAM = 0.015; // мягкая граница между половинами лица, в метрах
// Если левая и правая бровь работают наоборот, поставь true
const BROW_SWAP = false;

let debugCounter = 0;

// ---------- Помощники ----------
function clamp01(value) {
  return Math.min(1, Math.max(0, value));
}

function limit(value, max) {
  return Math.max(-max, Math.min(max, value));
}

// Обнуляет маленькие значения и вычитает порог из больших
function deadZone(value, dead) {
  if (Math.abs(value) <= dead) return 0;
  return value - Math.sign(value) * dead;
}

function fmt(value) {
  return value === undefined ? '-' : value.toFixed(2);
}

function vrmFlip() {
  return currentVrm && currentVrm.meta.metaVersion === '0' ? -1 : 1;
}

// ---------- Загрузчик VRM ----------
const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));

let currentVrm = null;
let faceMeshes = [];
let torsoBones = [];
let browSplit = false;

// Исходные позиции костей (для смещений по глубине)
let rest = {};
const POS_BONES = ['neck', 'head', 'leftShoulder', 'rightShoulder'];

function saveRestPositions(vrm) {
  rest = {};
  for (const name of POS_BONES) {
    const node = vrm.humanoid.getRawBoneNode(name);
    if (node) rest[name] = node.position.clone();
  }
}

// Оставляет в морфе только одну половину лица
function maskMorph(src, position, sideSign) {
  const out = new Float32Array(src.count * 3);
  for (let i = 0; i < src.count; i++) {
    const w = clamp01((position.getX(i) * sideSign + BROW_SEAM) / (2 * BROW_SEAM));
    out[i * 3] = src.getX(i) * w;
    out[i * 3 + 1] = src.getY(i) * w;
    out[i * 3 + 2] = src.getZ(i) * w;
  }
  return new THREE.BufferAttribute(out, 3);
}

// Создаёт морфы Fcl_BRW_*_L и Fcl_BRW_*_R из общих
function splitBrowMorphs(vrm) {
  browSplit = false;
  const leftSign = vrm.meta.metaVersion === '0' ? -1 : 1;
  const done = new Map();

  for (const mesh of faceMeshes) {
    const geometry = mesh.geometry;
    const morphPos = geometry.morphAttributes.position;
    const morphNor = geometry.morphAttributes.normal;
    const position = geometry.attributes.position;
    if (!morphPos || !position) continue;

    let added = done.get(geometry);
    if (!added) {
      added = {};
      for (const name of BROW_MORPHS) {
        const src = mesh.morphTargetDictionary[name];
        if (src === undefined) continue;

        for (const [suffix, sideSign] of [['_L', leftSign], ['_R', -leftSign]]) {
          const newIndex = morphPos.length;
          morphPos.push(maskMorph(morphPos[src], position, sideSign));
          if (morphNor && morphNor[src]) {
            morphNor.push(maskMorph(morphNor[src], position, sideSign));
          }
          added[name + suffix] = newIndex;
        }
      }
      done.set(geometry, added);
    }

    for (const key in added) {
      mesh.morphTargetDictionary[key] = added[key];
      browSplit = true;
    }
    while (mesh.morphTargetInfluences.length < morphPos.length) {
      mesh.morphTargetInfluences.push(0);
    }
  }
  console.log('Раздельные брови:', browSplit);
}

function loadVrmFromFile(file) {
  const url = URL.createObjectURL(file);

  loader.load(
    url,
    (gltf) => {
      const vrm = gltf.userData.vrm;

      if (currentVrm) {
        scene.remove(currentVrm.scene);
        VRMUtils.deepDispose(currentVrm.scene);
      }

      VRMUtils.rotateVRM0(vrm);

      currentVrm = vrm;
      scene.add(vrm.scene);
      URL.revokeObjectURL(url);

      const names = Object.keys(vrm.expressionManager.expressionMap);
      exprBox.textContent = 'Выражения модели (' + names.length + '):\n' + names.join(', ');

      faceMeshes = [];
      vrm.scene.traverse((object) => {
        if (
          object.isMesh &&
          object.morphTargetDictionary &&
          object.morphTargetDictionary['Fcl_MTH_Joy'] !== undefined
        ) {
          faceMeshes.push(object);
        }
      });

      // Кости туловища, которые есть у этой модели
      torsoBones = ['spine', 'chest', 'upperChest']
        .map((name) => vrm.humanoid.getNormalizedBoneNode(name))
        .filter(Boolean);

      splitBrowMorphs(vrm);
      saveRestPositions(vrm);

      console.log('Кусков лица с формами:', faceMeshes.length);
      console.log('Костей туловища:', torsoBones.length);
      console.log('Версия VRM:', vrm.meta.metaVersion);
    },
    undefined,
    (error) => {
      console.error(error);
      alert('Не удалось загрузить модель. Это точно файл .vrm?');
    }
  );
}

fileInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (file) {
    loadVrmFromFile(file);
  }
});

// ---------- Камера и трекинг ----------
let faceLandmarker = null;
let poseLandmarker = null;
let visionFileset = null;
let poseLoading = false;
let latestPose = null;
let latestShoulders = null;
let latestTorso = null;
let poseFrame = 0;

// Кнопка включает и выключает ТОЛЬКО руки
let armsOn = localStorage.getItem('armsOn') === '1';

function updateArmsButton() {
  armsButton.textContent = 'Руки: ' + (armsOn ? 'вкл' : 'выкл');
}
updateArmsButton();

armsButton.addEventListener('click', () => {
  armsOn = !armsOn;
  localStorage.setItem('armsOn', armsOn ? '1' : '0');
  updateArmsButton();
});

// Трекинг тела грузится всегда: он нужен для плеч и торса
async function loadPose() {
  if (!visionFileset || poseLandmarker || poseLoading) return;

  poseLoading = true;
  try {
    poseLandmarker = await PoseLandmarker.createFromOptions(visionFileset, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        delegate: 'GPU'
      },
      runningMode: 'VIDEO',
      numPoses: 1
    });
  } catch (error) {
    console.error('Не удалось запустить трекинг тела:', error);
  }
  poseLoading = false;
}

async function startTracking() {
  try {
    cameraButton.disabled = true;
    cameraButton.textContent = 'Загрузка...';

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720, frameRate: 60 }
    });
    video.srcObject = stream;
    await video.play();

    const fileset = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
    );
    visionFileset = fileset;

    faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU'
      },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true
    });

    cameraButton.textContent = 'Камера включена';

    await loadPose();
  } catch (error) {
    console.error(error);
    alert('Не получилось включить камеру или трекинг. Подробности в консоли (F12).');
    cameraButton.disabled = false;
    cameraButton.textContent = 'Включить камеру';
  }
}

cameraButton.addEventListener('click', startTracking);

calibrateButton.addEventListener('click', () => {
  calibrateButton.textContent = 'Расслабь лицо...';
  setTimeout(() => {
    calibration = { endTime: performance.now() + 2000, sums: {}, counts: {} };
    calibrateButton.textContent = 'Идёт замер...';
  }, 2000);
});

resetButton.addEventListener('click', () => {
  neutral = {};
  base = {};
  localStorage.removeItem('neutral');
});

// ---------- Выражения ----------
function setSmooth(name, target) {
  const expressions = currentVrm.expressionManager;
  const current = expressions.getValue(name) ?? 0;
  const next = current + (clamp01(target) - current) * 0.5;
  expressions.setValue(name, next);
}

function setMorphSmooth(name, target) {
  let current = 0;
  for (const mesh of faceMeshes) {
    const index = mesh.morphTargetDictionary[name];
    if (index !== undefined) {
      current = mesh.morphTargetInfluences[index] ?? 0;
      break;
    }
  }
  const next = current + (clamp01(target) - current) * 0.5;

  for (const mesh of faceMeshes) {
    const index = mesh.morphTargetDictionary[name];
    if (index !== undefined) {
      mesh.morphTargetInfluences[index] = next;
    }
  }
}

function level(shapes, name, range, dead = DEAD) {
  const base0 = neutral[name] ?? 0;
  const value = ((shapes[name] ?? 0) - base0) / range;
  return Math.max(0, value - dead) / (1 - dead);
}

function jawLevel(shapes) {
  const jawBase = Math.min(neutral.jawOpen ?? 0, JAW_MAX_BASE);
  const value = Math.max(0, (shapes.jawOpen ?? 0) - jawBase);

  jawPeak = Math.max(value, jawPeak * JAW_DECAY, JAW_MIN_PEAK);

  return clamp01((value - DEAD_JAW) / (jawPeak - DEAD_JAW));
}

// ---------- Повороты и смещения костей ----------
const worldQ = new THREE.Quaternion();
const localQ = new THREE.Quaternion();
const parentQ = new THREE.Quaternion();
const tmpEuler = new THREE.Euler();
const posTarget = new THREE.Vector3();

// Кость поворачивается так, как задано в мире, а программа сама учитывает
// повороты всех родителей (позвоночник, грудь, плечо...)
function applyWorldRotation(bone, qWorld, smooth) {
  if (!bone || !bone.parent) return;
  bone.parent.getWorldQuaternion(parentQ);
  localQ.copy(parentQ).invert().multiply(qWorld).multiply(currentVrm.scene.quaternion);
  bone.quaternion.slerp(localQ, smooth);
}

// Смещение отдельной кости. x, y, z в мировых осях (метры), z > 0 это к камере
function offsetBone(name, x, y, z) {
  if (!currentVrm) return;
  const node = currentVrm.humanoid.getRawBoneNode(name);
  const r = rest[name];
  if (!node || !r) return;

  const flip = vrmFlip();
  posTarget.set(r.x + x * flip, r.y + y, r.z + z * flip);
  node.position.lerp(posTarget, POS_SMOOTH);
}

// ---------- Голова ----------
const tmpMatrix = new THREE.Matrix4();
const tmpPosition = new THREE.Vector3();
const tmpQuaternion = new THREE.Quaternion();
const tmpScale = new THREE.Vector3();

function getHeadAngles(result) {
  const matrix = result.facialTransformationMatrixes?.[0];
  if (!matrix) return null;

  tmpMatrix.fromArray(matrix.data);
  tmpMatrix.decompose(tmpPosition, tmpQuaternion, tmpScale);
  tmpEuler.setFromQuaternion(tmpQuaternion, 'YXZ');

  return {
    yaw: tmpEuler.y,
    pitch: tmpEuler.x,
    roll: tmpEuler.z,
    x: tmpPosition.x,
    y: tmpPosition.y,
    z: tmpPosition.z
  };
}

// Считаем цели для головы, а кости двигает главный цикл
function updateHeadTarget(shapes) {
  const rawYaw = shapes.headYaw - (neutral.headYaw ?? 0);
  const rawPitch = shapes.headPitch - (neutral.headPitch ?? 0);
  const rawRoll = shapes.headRoll - (neutral.headRoll ?? 0);

  if (
    Math.abs(rawYaw) > MAX_VALID.yaw ||
    Math.abs(rawPitch) > MAX_VALID.pitch ||
    Math.abs(rawRoll) > MAX_VALID.roll
  ) {
    return;
  }

  headTarget.yaw = limit(rawYaw * YAW_SIGN, MAX_ANGLE.yaw);
  headTarget.pitch = limit(rawPitch * PITCH_SIGN, MAX_ANGLE.pitch);
  headTarget.roll = limit(rawRoll * ROLL_SIGN, MAX_ANGLE.roll);
}

function applyHeadBones(smooth) {
  const humanoid = currentVrm.humanoid;
  const neck = humanoid.getNormalizedBoneNode('neck');
  const head = humanoid.getNormalizedBoneNode('head');

  tmpEuler.set(
