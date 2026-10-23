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
    headTarget.pitch * NECK_SHARE,
    headTarget.yaw * NECK_SHARE,
    headTarget.roll * NECK_SHARE,
    'YXZ'
  );
  worldQ.setFromEuler(tmpEuler);
  applyWorldRotation(neck, worldQ, smooth);

  tmpEuler.set(headTarget.pitch, headTarget.yaw, headTarget.roll, 'YXZ');
  worldQ.setFromEuler(tmpEuler);
  applyWorldRotation(head, worldQ, smooth);
}

// Положение головы двигает только шею и голову, тело стоит на месте
function updateHeadOffset(shapes) {
  base.headX ??= shapes.headX;
  base.headY ??= shapes.headY;
  base.headZ ??= shapes.headZ;

  headOffset.x = limit((shapes.headX - base.headX) * HEAD_POS_SIGN.x * HEAD_POS_GAIN.x, HEAD_POS_MAX.x);
  headOffset.y = limit((shapes.headY - base.headY) * HEAD_POS_SIGN.y * HEAD_POS_GAIN.y, HEAD_POS_MAX.y);
  headOffset.z = limit((shapes.headZ - base.headZ) * HEAD_POS_SIGN.z * HEAD_POS_GAIN.z, HEAD_POS_MAX.z);
}

function applyHeadOffset() {
  offsetBone('neck', headOffset.x * NECK_SHARE, headOffset.y * NECK_SHARE, headOffset.z * NECK_SHARE);
  offsetBone('head', headOffset.x, headOffset.y, headOffset.z);
}

// ---------- Туловище (наклон вбок и поворот) ----------
function getTorsoAngles() {
  if (!latestPose) return null;

  const p = latestPose.image;
  const wz = latestPose.world;

  if ((p[11].visibility ?? 0) < POSE_VISIBLE || (p[12].visibility ?? 0) < POSE_VISIBLE) {
    return null;
  }

  const w = video.videoWidth;
  const h = video.videoHeight;

  return {
    roll: Math.atan2((p[11].y - p[12].y) * h, (p[11].x - p[12].x) * w),
    yaw: Math.atan2(wz[11].z - wz[12].z, wz[11].x - wz[12].x)
  };
}

function updateTorsoTarget() {
  torso.yaw = 0;
  torso.roll = 0;

  if (latestTorso && base.torsoRoll !== undefined && base.torsoYaw !== undefined) {
    const roll = deadZone(latestTorso.roll - base.torsoRoll, TORSO_DEAD);
    const yaw = deadZone(latestTorso.yaw - base.torsoYaw, TORSO_DEAD);
    torso.roll = limit(roll * TORSO_ROLL_SIGN, TORSO_MAX.roll);
    torso.yaw = limit(yaw * TORSO_YAW_SIGN, TORSO_MAX.yaw);
  }
}

function applyTorso() {
  torsoBones.forEach((bone, i) => {
    const share = (i + 1) / torsoBones.length;
    tmpEuler.set(0, torso.yaw * share, torso.roll * share, 'YXZ');
    worldQ.setFromEuler(tmpEuler);
    applyWorldRotation(bone, worldQ, TORSO_SMOOTH);
  });
}

// ---------- Плечи ----------
// Высота каждого плеча считается относительно средней высоты ушей
// (наклон головы почти не мешает). Глубина берётся из мировых координат тела
function getShoulderData() {
  if (!latestPose) return null;

  const p = latestPose.image;
  const wz = latestPose.world;

  if ((p[11].visibility ?? 0) < POSE_VISIBLE || (p[12].visibility ?? 0) < POSE_VISIBLE) {
    return null;
  }

  let sum = 0;
  let n = 0;
  for (const i of [7, 8]) {
    if ((p[i].visibility ?? 0) > POSE_VISIBLE) {
      sum += p[i].y;
      n++;
    }
  }
  if (n === 0) return null;
  const refY = sum / n;

  const w = video.videoWidth;
  const h = video.videoHeight;
  const width = Math.hypot((p[11].x - p[12].x) * w, (p[11].y - p[12].y) * h);
  if (width < 20) return null;

  return {
    left: ((p[11].y - refY) * h) / width,
    right: ((p[12].y - refY) * h) / width,
    depthLeft: wz[11].z,
    depthRight: wz[12].z
  };
}

function applyShoulder(side, boneName, gap, baseGap, depth, baseDepth) {
  const bone = currentVrm.humanoid.getNormalizedBoneNode(boneName);
  if (!bone) return;

  // Подъём и опускание: поворот ЛОКАЛЬНЫЙ, чтобы плечо ехало вместе с торсом
  let angle = 0;
  if (gap !== undefined && baseGap !== undefined) {
    angle = limit(deadZone(baseGap - gap, SHOULDER_DEAD) * SHOULDER_GAIN, SHOULDER_MAX);
  }

  // VRM0 повёрнут на 180° по Y, поэтому ось Z локально зеркальная
  tmpEuler.set(0, 0, side * angle * SHOULDER_SIGN * vrmFlip(), 'YXZ');
  worldQ.setFromEuler(tmpEuler);
  bone.quaternion.slerp(worldQ, SHOULDER_SMOOTH);

  // Глубина этого плеча
  let dz = 0;
  if (depth !== undefined && baseDepth !== undefined) {
    dz = limit(
      deadZone((depth - baseDepth) * SHOULDER_DEPTH_SIGN, SHOULDER_DEPTH_DEAD) * SHOULDER_DEPTH_GAIN,
      SHOULDER_DEPTH_MAX
    );
  }
  offsetBone(boneName, 0, 0, dz);
}

function applyShoulders() {
  const s = latestShoulders;
  applyShoulder(1, 'leftShoulder', s?.right, base.gapRight, s?.depthRight, base.depthRight);
  applyShoulder(-1, 'rightShoulder', s?.left, base.gapLeft, s?.depthLeft, base.depthLeft);
}

// ---------- Руки ----------
const armRest = new THREE.Vector3();
const armWant = new THREE.Vector3();
const dirUpper = new THREE.Vector3();
const dirLower = new THREE.Vector3();
const upperQ = new THREE.Quaternion();
const lowerQ = new THREE.Quaternion();

function setDirection(from, to, target) {
  target.set(
    (to.x - from.x) * ARM_SIGN.x,
    (to.y - from.y) * ARM_SIGN.y,
    (to.z - from.z) * ARM_SIGN.z
  );
}

function quaternionFromDirection(side, dir, out) {
  if (dir.lengthSq() < 0.000001) {
    out.identity();
    return;
  }
  armRest.set(side, 0, 0);
  armWant.copy(dir).normalize();
  out.setFromUnitVectors(armRest, armWant);
}

function applyArm(side, shoulderId, elbowId, wristId) {
  const humanoid = currentVrm.humanoid;
  const upperBone = humanoid.getNormalizedBoneNode(side === 1 ? 'leftUpperArm' : 'rightUpperArm');
  const lowerBone = humanoid.getNormalizedBoneNode(side === 1 ? 'leftLowerArm' : 'rightLowerArm');
  if (!upperBone || !lowerBone) return;

  let upperVisible = false;
  let lowerVisible = false;

  // Если руки выключены, они возвращаются в спокойную позу
  if (latestPose && armsOn) {
    const image = latestPose.image;
    const world = latestPose.world;

    upperVisible =
      (image[shoulderId].visibility ?? 0) > POSE_VISIBLE &&
      (image[elbowId].visibility ?? 0) > POSE_VISIBLE;
    lowerVisible = upperVisible && (image[wristId].visibility ?? 0) > POSE_VISIBLE;

    if (upperVisible) {
      setDirection(world[shoulderId], world[elbowId], dirUpper);
    }
    if (lowerVisible) {
      setDirection(world[elbowId], world[wristId], dirLower);
    }
  }

  if (!upperVisible) {
    dirUpper.set(side * ARM_REST_OUT, -1, 0);
  }
  if (!lowerVisible) {
    dirLower.copy(dirUpper);
  }

  quaternionFromDirection(side, dirUpper, upperQ);
  quaternionFromDirection(side, dirLower, lowerQ);

  // Руки абсолютные: учитывают и торс, и плечо
  applyWorldRotation(upperBone, upperQ, ARM_SMOOTH);
  applyWorldRotation(lowerBone, lowerQ, ARM_SMOOTH);
}

function applyArms() {
  applyArm(1, 12, 14, 16);
  applyArm(-1, 11, 13, 15);
}

function updatePose() {
  poseFrame++;
  if (poseFrame % POSE_EVERY !== 0) return;

  const result = poseLandmarker.detectForVideo(video, performance.now());
  if (result.landmarks.length > 0 && result.worldLandmarks.length > 0) {
    latestPose = { image: result.landmarks[0], world: result.worldLandmarks[0] };
    latestShoulders = getShoulderData();
    latestTorso = getTorsoAngles();

    if (latestShoulders) {
      base.gapLeft ??= latestShoulders.left;
      base.gapRight ??= latestShoulders.right;
      base.depthLeft ??= latestShoulders.depthLeft;
      base.depthRight ??= latestShoulders.depthRight;
    }
    if (latestTorso) {
      base.torsoRoll ??= latestTorso.roll;
      base.torsoYaw ??= latestTorso.yaw;
    }
  } else {
    latestPose = null;
    latestShoulders = null;
    latestTorso = null;
  }
}

// ---------- Применяем лицо к модели ----------
function applyFace(result) {
  const shapes = {};
  for (const item of result.faceBlendshapes[0].categories) {
    shapes[item.categoryName] = item.score;
  }

  const headAngles = getHeadAngles(result);
  if (headAngles) {
    shapes.headYaw = headAngles.yaw;
    shapes.headPitch = headAngles.pitch;
    shapes.headRoll = headAngles.roll;
    shapes.headX = headAngles.x;
    shapes.headY = headAngles.y;
    shapes.headZ = headAngles.z;
  }
  if (latestShoulders) {
    shapes.gapLeft = latestShoulders.left;
    shapes.gapRight = latestShoulders.right;
    shapes.depthLeft = latestShoulders.depthLeft;
    shapes.depthRight = latestShoulders.depthRight;
  }
  if (latestTorso) {
    shapes.torsoRoll = latestTorso.roll;
    shapes.torsoYaw = latestTorso.yaw;
  }

  if (calibration) {
    for (const name in shapes) {
      calibration.sums[name] = (calibration.sums[name] ?? 0) + shapes[name];
      calibration.counts[name] = (calibration.counts[name] ?? 0) + 1;
    }

    if (performance.now() > calibration.endTime) {
      for (const name in calibration.sums) {
        neutral[name] = calibration.sums[name] / calibration.counts[name];
      }
      // Данные тела уходят в base и не сохраняются
      for (const key of BODY_KEYS) {
        if (neutral[key] !== undefined) {
          base[key] = neutral[key];
          delete neutral[key];
        }
      }
      localStorage.setItem('neutral', JSON.stringify(neutral));
      calibration = null;
      calibrateButton.textContent = 'Калибровка';
    }
  }

  debugCounter++;
  if (debugCounter % 10 === 0) {
    const names = [
      'eyeBlinkLeft', 'eyeBlinkRight', 'jawOpen',
      'mouthPucker', 'mouthSmileLeft', 'mouthSmileRight',
      'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
      'browDownLeft', 'browDownRight',
      'headYaw', 'headPitch', 'headRoll'
    ];
    debugBox.textContent =
      names
        .map((n) => n + ': сейчас ' + (shapes[n] ?? 0).toFixed(2) + ', покой ' + (neutral[n] ?? 0).toFixed(2))
        .join('\n') +
      '\nпик рта: ' + jawPeak.toFixed(2) +
      '\nтело: ' + (latestPose ? 'видно' : 'не видно') +
      '\nраздельные брови: ' + (browSplit ? 'да' : 'нет') +
      '\nкостей торса: ' + torsoBones.length +
      '\nдистанция: ' + (shapes.headZ ?? 0).toFixed(1) + ' см' +
      '\nплечи зазор L/R: ' + fmt(latestShoulders?.left) + ' / ' + fmt(latestShoulders?.right) +
      ' (покой ' + fmt(base.gapLeft) + ' / ' + fmt(base.gapRight) + ')' +
      '\nплечи глубина L/R: ' + fmt(latestShoulders?.depthLeft) + ' / ' + fmt(latestShoulders?.depthRight) +
      '\nкорпус наклон/поворот: ' + fmt(latestTorso?.roll) + ' / ' + fmt(latestTorso?.yaw) +
      ' (покой ' + fmt(base.torsoRoll) + ' / ' + fmt(base.torsoYaw) + ')';
  }

  setSmooth('blinkLeft', level(shapes, 'eyeBlinkLeft', RANGE.blink));
  setSmooth('blinkRight', level(shapes, 'eyeBlinkRight', RANGE.blink));
  setSmooth('aa', jawLevel(shapes));
  setSmooth('ou', level(shapes, 'mouthPucker', RANGE.pucker));

  const smile =
    (level(shapes, 'mouthSmileLeft', RANGE.smile) +
      level(shapes, 'mouthSmileRight', RANGE.smile)) / 2;
  setMorphSmooth('Fcl_MTH_Joy', smile);

  const browInner = level(shapes, 'browInnerUp', RANGE.brow);

  if (browSplit) {
    // Каждая бровь живёт своей жизнью
    for (const side of ['Left', 'Right']) {
      const outer = level(shapes, 'browOuterUp' + side, RANGE.brow);
      const down = level(shapes, 'browDown' + side, RANGE.brow);
      const sorrow = clamp01((browInner - outer) * SORROW_GAIN);
      const suffix = (side === 'Left') !== BROW_SWAP ? '_L' : '_R';

      setMorphSmooth('Fcl_BRW_Surprised' + suffix, outer);
      setMorphSmooth('Fcl_BRW_Angry' + suffix, down);
      setMorphSmooth('Fcl_BRW_Sorrow' + suffix, sorrow);
    }
  } else {
    // Запасной вариант: симметричные брови
    const browOuter =
      (level(shapes, 'browOuterUpLeft', RANGE.brow) +
        level(shapes, 'browOuterUpRight', RANGE.brow)) / 2;
    const browDown =
      (level(shapes, 'browDownLeft', RANGE.brow) +
        level(shapes, 'browDownRight', RANGE.brow)) / 2;
    const browSorrow = clamp01((browInner - browOuter) * SORROW_GAIN);

    setMorphSmooth('Fcl_BRW_Surprised', browOuter);
    setMorphSmooth('Fcl_BRW_Angry', browDown);
    setMorphSmooth('Fcl_BRW_Sorrow', browSorrow);
  }

  if (headAngles) {
    updateHeadTarget(shapes);
    updateHeadOffset(shapes);
  }
}

// ---------- Главный цикл ----------
const clock = new THREE.Clock();
let lastVideoTime = -1;

function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();

  if (faceLandmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    // Сначала тело, чтобы плечи и торс были свежими, когда считается лицо
    if (poseLandmarker) {
      updatePose();
    }

    const result = faceLandmarker.detectForVideo(video, performance.now());
    if (currentVrm && result.faceBlendshapes.length > 0) {
      lastFaceTime = performance.now();
      applyFace(result);
    }
  }

  if (currentVrm) {
    const faceLost = faceLandmarker && performance.now() - lastFaceTime > LOST_DELAY;
    if (faceLost) {
      headTarget.yaw = 0;
      headTarget.pitch = 0;
      headTarget.roll = 0;
      headOffset.x = 0;
      headOffset.y = 0;
      headOffset.z = 0;
    }

    // Порядок важен: сначала туловище, потом то, что к нему прикреплено
    updateTorsoTarget();
    applyTorso();
    applyShoulders();
    applyHeadBones(faceLost ? RELAX_SMOOTH : HEAD_SMOOTH);
    applyHeadOffset();
    applyArms();

    currentVrm.update(delta);
  }

  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
