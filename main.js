import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import {
  FaceLandmarker,
  PoseLandmarker,
  HandLandmarker,
  FilesetResolver
} from '@mediapipe/tasks-vision';

// ---------- Общие переменные ----------
let currentVrm = null;
let faceMeshes = [];
let torsoBones = [];
let browSplit = false;

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
const armsButton = document.getElementById('bodyButton'); // кнопка рук (руки, кисти, пальцы)
const panel = document.getElementById('panel');

// Кнопка зеркала создаётся здесь, index.html менять не нужно
const mirrorButton = document.createElement('button');
armsButton.insertAdjacentElement('afterend', mirrorButton);

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
const ARM_SMOOTH = 0.2;
const ARM_REST_OUT = 0.25;
const POSE_EVERY = 2;

// Кисти и пальцы
const WRIST_ON = true;          // false = кисти не поворачиваются, только пальцы
const HAND_SMOOTH = 0.3;        // плавность поворота кисти
const HAND_EVERY = 3;           // трекинг кистей раз в N кадров
// Если у кисти видна ладонь вместо тыльной стороны (или наоборот), поменяй true/false
const HAND_FLIP = { left: true, right: true };
const HAND_MAX_BEND = 1.7;      // максимум поворота кисти относительно предплечья, радианы
const FINGER_SIGN = 1;          // если пальцы сгибаются в обратную сторону, поставь -1
const THUMB_SIGN = 1;           // то же для большого пальца
const FINGER_GAIN = 1.1;        // сила сгиба
const FINGER_DEAD = 0.08;       // мёртвая зона сгиба, радианы
const FINGER_MAX = { proximal: 1.5, intermediate: 1.9, distal: 1.3 }; // максимум сгиба, радианы
const FINGER_REST = 0.12;       // лёгкий естественный изгиб, когда кисть не видна
const FINGER_SMOOTH = 0.4;

// Плечи: подъём и опускание (у каждого плеча отдельно)
const SHOULDER_SIGN = 1;
const SHOULDER_GAIN = 3;
const SHOULDER_DEAD = 0.03;
const SHOULDER_MAX = 0.5;
const SHOULDER_SMOOTH = 0.2;

// Плечи: глубина (вперёд и назад, у каждого плеча отдельно)
const SHOULDER_DEPTH_SIGN = -1; // если плечо уходит не в ту сторону, поставь 1
const SHOULDER_DEPTH_GAIN = 1;
const SHOULDER_DEPTH_DEAD = 0.02;
const SHOULDER_DEPTH_MAX = 0.15;

// Туловище: наклон вбок и поворот (по плечам). Вперёд и назад не наклоняется
const TORSO_ROLL_SIGN = 1;
const TORSO_YAW_SIGN = -1;
const TORSO_DEAD = 0.05;
const TORSO_MAX = { yaw: 0.8, roll: 0.6 };
const TORSO_SMOOTH = 0.15;
const torso = { yaw: 0, roll: 0 };

// Фильтр One Euro: гасит дрожание, когда стоишь неподвижно
// Меньше FILTER_MIN_CUTOFF = плавнее, но с задержкой
// Больше FILTER_BETA = быстрее реагирует на резкие движения
const FILTER_MIN_CUTOFF = 0.5;
const FILTER_BETA = 5;

// Взгляд (зрачки)
const GAZE_YAW_SIGN = 1;     // если глаза смотрят в другую сторону по горизонтали, поставь -1
const GAZE_PITCH_SIGN = 1;   // если глаза смотрят в другую сторону по вертикали, поставь -1
const GAZE_DEAD = { x: 0.03, y: 0.02 };
const GAZE_MIN_PEAK = { x: 0.2, y: 0.1 };   // минимальный «упор» взгляда, меньше не делим
const GAZE_MAX_PEAK = { x: 0.8, y: 0.6 };
const GAZE_PEAK_DECAY = 0.9995;             // как быстро забывается старый максимум
const GAZE_MAX = { yaw: 90, pitch: 90 };    // градусы на входе lookAt (ход глаз ограничивает сама модель)
const GAZE_SMOOTH = 0.5;
const GAZE_FILTER_CUTOFF = 1.5;
const GAZE_BLINK_COMP = 0.8;                // взгляд вниз не должен закрывать веки
const gazeTarget = { yaw: 0, pitch: 0 };
const gaze = { yaw: 0, pitch: 0 };
const gazePeaks = { xp: GAZE_MIN_PEAK.x, xn: GAZE_MIN_PEAK.x, yp: GAZE_MIN_PEAK.y, yn: GAZE_MIN_PEAK.y };

// Брови
const SORROW_GAIN = 0.7;
const BROW_UP_GAIN = 1.3;    // сила подъёма бровей
const BROW_DOWN_GAIN = 1.8;  // сила опускания (у MediaPipe она обычно слабее)
const BROW_ASYM = 0.9;       // недоумение: чем выше одна бровь, тем сильнее опускается другая (0 = выключить)
const BROW_ASYM_DEAD = 0.2;  // насколько бровь должна быть выше, чтобы вторая начала опускаться
const BROW_MORPHS = ['Fcl_BRW_Surprised', 'Fcl_BRW_Angry', 'Fcl_BRW_Sorrow'];
const BROW_SEAM = 0.015;
// Если левая и правая бровь работают наоборот, поставь true
const BROW_SWAP = false;

// Стороны лица (веки, брови). Если левый глаз модели моргает правым, поставь true
const FACE_SWAP = false;

let debugCounter = 0;
let browDebug = { L: [0, 0], R: [0, 0] };
let trackingErrorShown = false;

// ---------- Зеркало ----------
// Вкл: модель повторяет твоё отражение (как превью камеры)
// Выкл: правая рука человека управляет правой рукой модели
let mirror = localStorage.getItem('mirror') !== '0';

function mir() {
  return mirror ? 1 : -1;
}

function updateMirrorButton() {
  mirrorButton.textContent = 'Зеркало: ' + (mirror ? 'вкл' : 'выкл');
  video.style.transform = mirror ? 'scaleX(-1)' : 'none';
}
updateMirrorButton();

mirrorButton.addEventListener('click', () => {
  mirror = !mirror;
  localStorage.setItem('mirror', mirror ? '1' : '0');
  updateMirrorButton();
  latestHands = { left: null, right: null };
  resetFilters('hL');
  resetFilters('hR');
});

// Какая сторона лица MediaPipe управляет стороной модели
function mpSide(modelSide, extraSwap = false) {
  const flip = (!mirror !== FACE_SWAP) !== extraSwap;
  if (!flip) return modelSide;
  return modelSide === 'Left' ? 'Right' : 'Left';
}

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

// ---------- Фильтр One Euro ----------
function filterAlpha(cutoff, dt) {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}

class OneEuro {
  constructor(minCutoff, beta) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = 1;
    this.x = null;
    this.dx = 0;
    this.t = 0;
  }

  filter(value, time) {
    if (this.x === null) {
      this.x = value;
      this.t = time;
      return value;
    }
    const dt = Math.max((time - this.t) / 1000, 0.001);
    const rawSpeed = (value - this.x) / dt;
    this.dx += filterAlpha(this.dCutoff, dt) * (rawSpeed - this.dx);

    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    this.x += filterAlpha(cutoff, dt) * (value - this.x);
    this.t = time;
    return this.x;
  }
}

// Фильтры точек тела и кистей
let landmarkFilters = {};

function getFilter(key) {
  if (!landmarkFilters[key]) {
    landmarkFilters[key] = new OneEuro(FILTER_MIN_CUTOFF, FILTER_BETA);
  }
  return landmarkFilters[key];
}

function smoothLandmarks(list, kind, time) {
  return list.map((point, i) => ({
    x: getFilter(kind + i + 'x').filter(point.x, time),
    y: getFilter(kind + i + 'y').filter(point.y, time),
    z: getFilter(kind + i + 'z').filter(point.z, time),
    visibility: point.visibility
  }));
}

function resetFilters(prefix) {
  for (const key in landmarkFilters) {
    if (key.startsWith(prefix)) delete landmarkFilters[key];
  }
}

// Фильтры взгляда
const gazeFilterX = new OneEuro(GAZE_FILTER_CUTOFF, FILTER_BETA);
const gazeFilterY = new OneEuro(GAZE_FILTER_CUTOFF, FILTER_BETA);

// ---------- Загрузчик VRM ----------
const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));

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
        const dict = object.isMesh && object.morphTargetDictionary;
        if (
          dict &&
          (dict['Fcl_MTH_Joy'] !== undefined ||
            BROW_MORPHS.some((name) => dict[name] !== undefined))
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

      gaze.yaw = 0;
      gaze.pitch = 0;

      const fingerCount = ['leftIndexProximal', 'rightIndexProximal', 'leftThumbProximal']
        .filter((name) => vrm.humanoid.getNormalizedBoneNode(name)).length;

      console.log('Кусков лица с формами:', faceMeshes.length);
      console.log('Костей туловища:', torsoBones.length);
      console.log('Кости пальцев найдены:', fingerCount > 0 ? 'да' : 'нет');
      console.log('Управление взглядом:', vrm.lookAt ? 'есть' : 'нет');
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
let handLandmarker = null;
let visionFileset = null;
let poseLoading = false;
let handLoading = false;
let latestPose = null;
let latestShoulders = null;
let latestTorso = null;
let latestHands = { left: null, right: null }; // left/right = стороны МОДЕЛИ
let poseFrame = 0;
let handFrame = 0;

// Кнопка включает и выключает руки, кисти и пальцы
let armsOn = localStorage.getItem('armsOn') === '1';

function updateArmsButton() {
  armsButton.textContent = 'Руки: ' + (armsOn ? 'вкл' : 'выкл');
}
updateArmsButton();

armsButton.addEventListener('click', async () => {
  armsOn = !armsOn;
  localStorage.setItem('armsOn', armsOn ? '1' : '0');
  updateArmsButton();

  if (armsOn) {
    await loadHands();
  } else {
    unloadHands();
  }
});

// Создаёт трекер на GPU, а если не вышло, то на CPU
async function createTracker(TrackerClass, modelUrl, extra) {
  for (const delegate of ['GPU', 'CPU']) {
    try {
      return await TrackerClass.createFromOptions(visionFileset, {
        baseOptions: { modelAssetPath: modelUrl, delegate },
        runningMode: 'VIDEO',
        ...extra
      });
    } catch (error) {
      console.warn('Трекер не запустился на ' + delegate + ':', error);
    }
  }
  return null;
}

// Трекинг тела грузится всегда: он нужен для плеч и торса
async function loadPose() {
  if (!visionFileset || poseLandmarker || poseLoading) return;

  poseLoading = true;
  poseLandmarker = await createTracker(
    PoseLandmarker,
    'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
    { numPoses: 1 }
  );
  if (!poseLandmarker) console.error('Не удалось запустить трекинг тела');
  poseLoading = false;
}

// Трекинг кистей грузится только когда включены руки
async function loadHands() {
  if (!visionFileset || handLandmarker || handLoading) return;

  handLoading = true;
  armsButton.textContent = 'Загрузка кистей...';
  handLandmarker = await createTracker(
    HandLandmarker,
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    { numHands: 2 }
  );
  if (!handLandmarker) console.error('Не удалось запустить трекинг кистей');
  handLoading = false;
  updateArmsButton();
}

function unloadHands() {
  if (handLandmarker) {
    handLandmarker.close();
    handLandmarker = null;
  }
  latestHands = { left: null, right: null };
  resetFilters('hL');
  resetFilters('hR');
}

function stopCamera() {
  const stream = video.srcObject;
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    video.srcObject = null;
  }
}

// Камера запрашивается каскадом: от лучшего режима к самому простому
async function openCamera() {
  const attempts = [
    { video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60 } } },
    { video: { width: { ideal: 640 }, height: { ideal: 480 } } },
    { video: true }
  ];

  let lastError = null;
  for (const constraints of attempts) {
    try {
      return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (error) {
      lastError = error;
      // Запрет доступа повторять бессмысленно
      if (error.name === 'NotAllowedError' || error.name === 'SecurityError') break;
      await new Promise((resolve) => setTimeout(resolve, 300));
    }
  }
  throw lastError;
}

function cameraErrorMessage(error) {
  switch (error?.name) {
    case 'NotReadableError':
    case 'AbortError':
      return 'Не удалось занять камеру: её, скорее всего, использует другое приложение или вкладка (OBS, Zoom, Discord, Telegram, другая вкладка). Закрой их и нажми кнопку ещё раз. Если не помогло, перезагрузи страницу или переподключи камеру.';
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Нет доступа к камере. Разреши камеру для этого сайта в настройках браузера (значок замка рядом с адресом) и перезагрузи страницу.';
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'Камера не найдена. Проверь, что она подключена и включена в системе.';
    default:
      return 'Не получилось включить камеру или трекинг. Подробности в консоли (F12).';
  }
}

async function startTracking() {
  try {
    cameraButton.disabled = true;
    cameraButton.textContent = 'Загрузка...';

    stopCamera(); // на случай прошлой неудачной попытки

    const stream = await openCamera();
    video.srcObject = stream;
    await video.play();

    visionFileset = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
    );

    faceLandmarker = await createTracker(
      FaceLandmarker,
      'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      {
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true
      }
    );
    if (!faceLandmarker) throw new Error('Не удалось запустить трекинг лица');

    cameraButton.textContent = 'Камера включена';

    await loadPose();
    if (armsOn) {
      await loadHands();
    }
  } catch (error) {
    console.error(error);
    stopCamera(); // освобождаем камеру, чтобы повторная попытка сработала
    alert(cameraErrorMessage(error));
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
  gazePeaks.xp = gazePeaks.xn = GAZE_MIN_PEAK.x;
  gazePeaks.yp = gazePeaks.yn = GAZE_MIN_PEAK.y;
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

// ---------- Брови (каждая сама по себе) ----------
function applyBrows(shapes) {
  const innerUp = level(shapes, 'browInnerUp', RANGE.brow);

  if (!browSplit) {
    // Запасной вариант: симметричные брови
    const outerAvg =
      (level(shapes, 'browOuterUpLeft', RANGE.brow) +
        level(shapes, 'browOuterUpRight', RANGE.brow)) / 2;
    const downAvg =
      (level(shapes, 'browDownLeft', RANGE.brow) +
        level(shapes, 'browDownRight', RANGE.brow)) / 2;
    const sorrowAvg = clamp01((innerUp - outerAvg) * SORROW_GAIN);

    setMorphSmooth('Fcl_BRW_Surprised', outerAvg);
    setMorphSmooth('Fcl_BRW_Angry', downAvg);
    setMorphSmooth('Fcl_BRW_Sorrow', sorrowAvg);
    return;
  }

  // Сначала считаем подъём и опускание каждой брови отдельно
  const sides = {};
  for (const modelSide of ['Left', 'Right']) {
    const mp = mpSide(modelSide, BROW_SWAP);
    sides[modelSide] = {
      up: clamp01(level(shapes, 'browOuterUp' + mp, RANGE.brow) * BROW_UP_GAIN),
      down: clamp01(level(shapes, 'browDown' + mp, RANGE.brow) * BROW_DOWN_GAIN)
    };
  }

  // Ложная «грусть», когда одна бровь поднята, не нужна
  const maxUp = Math.max(sides.Left.up, sides.Right.up);
  const sorrowBase = clamp01((innerUp - maxUp) * SORROW_GAIN);

  for (const modelSide of ['Left', 'Right']) {
    const mine = sides[modelSide];
    const other = sides[modelSide === 'Left' ? 'Right' : 'Left'];

    // Недоумение: если другая бровь заметно выше, эта опускается
    const asym = Math.max(0, other.up - mine.up - BROW_ASYM_DEAD) * BROW_ASYM;
    const down = clamp01(Math.max(mine.down, asym));

    // Вверх и вниз на одной брови не работают одновременно: побеждает сильнее
    const net = mine.up - down;
    const up = Math.max(0, net);
    const frown = Math.max(0, -net);
    const sorrow = sorrowBase * (1 - frown);

    const suffix = modelSide === 'Left' ? '_L' : '_R';
    setMorphSmooth('Fcl_BRW_Surprised' + suffix, up);
    setMorphSmooth('Fcl_BRW_Angry' + suffix, frown);
    setMorphSmooth('Fcl_BRW_Sorrow' + suffix, sorrow);

    browDebug[modelSide === 'Left' ? 'L' : 'R'] = [up, frown];
  }
}

// ---------- Взгляд (зрачки) ----------
// Для каждого из четырёх направлений диапазон подстраивается сам:
// вверх у MediaPipe значения заметно меньше, чем вправо или влево
function normGaze(value, axis) {
  const key = axis + (value >= 0 ? 'p' : 'n');
  const decayed = gazePeaks[key] * GAZE_PEAK_DECAY;
  gazePeaks[key] = Math.min(
    GAZE_MAX_PEAK[axis],
    Math.max(Math.abs(value), decayed, GAZE_MIN_PEAK[axis])
  );
  return limit(value / gazePeaks[key], 1);
}

function updateGazeTarget(shapes, time) {
  const v = (name) => (shapes[name] ?? 0) - (neutral[name] ?? 0);

  const horizontal =
    (v('eyeLookOutLeft') + v('eyeLookInRight') - v('eyeLookInLeft') - v('eyeLookOutRight')) / 2;
  const vertical =
    (v('eyeLookUpLeft') + v('eyeLookUpRight') - v('eyeLookDownLeft') - v('eyeLookDownRight')) / 2;

  const hx = deadZone(gazeFilterX.filter(horizontal, time), GAZE_DEAD.x);
  const hy = deadZone(gazeFilterY.filter(vertical, time), GAZE_DEAD.y);

  const nx = normGaze(hx, 'x');
  const ny = normGaze(hy, 'y');

  gazeTarget.yaw = nx * GAZE_MAX.yaw * GAZE_YAW_SIGN * mir();
  gazeTarget.pitch = ny * GAZE_MAX.pitch * GAZE_PITCH_SIGN;
}

function applyGaze(faceLost) {
  const lookAt = currentVrm.lookAt;
  if (!lookAt) return;

  if (faceLost) {
    gazeTarget.yaw = 0;
    gazeTarget.pitch = 0;
  }

  gaze.yaw += (gazeTarget.yaw - gaze.yaw) * GAZE_SMOOTH;
  gaze.pitch += (gazeTarget.pitch - gaze.pitch) * GAZE_SMOOTH;

  lookAt.yaw = gaze.yaw;
  lookAt.pitch = gaze.pitch;
}

// ---------- Повороты и смещения костей ----------
const worldQ = new THREE.Quaternion();
const localQ = new THREE.Quaternion();
const parentQ = new THREE.Quaternion();
const identityQ = new THREE.Quaternion();
const tmpEuler = new THREE.Euler();
const posTarget = new THREE.Vector3();

function isFiniteQuaternion(q) {
  return Number.isFinite(q.x + q.y + q.z + q.w);
}

// Кость поворачивается так, как задано в мире, а программа сама учитывает
// повороты всех родителей (позвоночник, грудь, плечо...)
function applyWorldRotation(bone, qWorld, smooth) {
  if (!bone || !bone.parent) return;
  if (!isFiniteQuaternion(qWorld)) return;
  bone.parent.getWorldQuaternion(parentQ);
  localQ.copy(parentQ).invert().multiply(qWorld).multiply(currentVrm.scene.quaternion);
  if (!isFiniteQuaternion(localQ)) return;
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

  headTarget.yaw = limit(rawYaw * YAW_SIGN * mir(), MAX_ANGLE.yaw);
  headTarget.pitch = limit(rawPitch * PITCH_SIGN, MAX_ANGLE.pitch);
  headTarget.roll = limit(rawRoll * ROLL_SIGN * mir(), MAX_ANGLE.roll);
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

  headOffset.x = limit((shapes.headX - base.headX) * HEAD_POS_SIGN.x * mir() * HEAD_POS_GAIN.x, HEAD_POS_MAX.x);
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
    torso.roll = limit(roll * TORSO_ROLL_SIGN * mir(), TORSO_MAX.roll);
    torso.yaw = limit(yaw * TORSO_YAW_SIGN * mir(), TORSO_MAX.yaw);
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

// Какое плечо человека ('left' или 'right') управляет плечом модели
function shoulderSource(modelSide) {
  const isLeft = modelSide === 'left';
  return (mirror ? !isLeft : isLeft) ? 'left' : 'right';
}

function applyShoulders() {
  const s = latestShoulders;
  const list = [
    [1, 'leftShoulder', 'left'],
    [-1, 'rightShoulder', 'right']
  ];

  for (const [side, boneName, modelSide] of list) {
    const src = shoulderSource(modelSide);
    const K = src === 'left' ? 'Left' : 'Right';
    applyShoulder(side, boneName, s?.[src], base['gap' + K], s?.['depth' + K], base['depth' + K]);
  }
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
    (to.x - from.x) * ARM_SIGN.x * mir(),
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
  if (mirror) {
    // Левая рука модели повторяет правую руку человека (как отражение)
    applyArm(1, 12, 14, 16);
    applyArm(-1, 11, 13, 15);
  } else {
    // Левая рука модели повторяет левую руку человека
    applyArm(1, 11, 13, 15);
    applyArm(-1, 12, 14, 16);
  }
}

// ---------- Кисти и пальцы ----------
// Индексы точек MediaPipe Hands: 0 запястье, 1-4 большой, 5-8 указательный,
// 9-12 средний, 13-16 безымянный, 17-20 мизинец
const FINGERS = [
  ['Index', 5, 6, 7, 8],
  ['Middle', 9, 10, 11, 12],
  ['Ring', 13, 14, 15, 16],
  ['Little', 17, 18, 19, 20]
];

const bendA = new THREE.Vector3();
const bendB = new THREE.Vector3();
const fingerQ = new THREE.Quaternion();
const handF = new THREE.Vector3();
const handS = new THREE.Vector3();
const handU = new THREE.Vector3();
const handBasis = new THREE.Matrix4();
const handDetectedQ = new THREE.Quaternion();
const handRestQ = new THREE.Quaternion();
const forearmQ = new THREE.Quaternion();
const sceneInvQ = new THREE.Quaternion();
const handRelQ = new THREE.Quaternion();
const handLimitQ = new THREE.Quaternion();

// Угол сгиба в точке b между отрезками a-b и b-c (0 = палец прямой)
function bendAt(a, b, c) {
  bendA.set(b.x - a.x, b.y - a.y, b.z - a.z);
  bendB.set(c.x - b.x, c.y - b.y, c.z - b.z);
  if (bendA.lengthSq() < 1e-10 || bendB.lengthSq() < 1e-10) return 0;
  return bendA.angleTo(bendB);
}

function curlAmount(angle, max) {
  return Math.min(max, deadZone(Math.max(0, angle), FINGER_DEAD) * FINGER_GAIN);
}

function setFingerBone(name, y, z) {
  const bone = currentVrm.humanoid.getNormalizedBoneNode(name);
  if (!bone) return;
  tmpEuler.set(0, y, z, 'XYZ');
  fingerQ.setFromEuler(tmpEuler);
  bone.quaternion.slerp(fingerQ, FINGER_SMOOTH);
}

// Поворот кисти в мире: по направлению пальцев и плоскости ладони
function handWorldQuaternion(side, w, out) {
  setDirection(w[0], w[9], handF);   // от запястья к основанию среднего пальца
  setDirection(w[5], w[17], handS);  // от указательного к мизинцу
  if (handF.lengthSq() < 1e-8 || handS.lengthSq() < 1e-8) return false;

  handF.normalize();
  handS.addScaledVector(handF, -handS.dot(handF));
  if (handS.lengthSq() < 1e-8) return false;
  handS.normalize();

  // Разворот на 180° вокруг оси пальцев: ладонь и тыльная сторона меняются местами
  if (HAND_FLIP[side === 1 ? 'left' : 'right']) handS.negate();

  handU.crossVectors(handS, handF);

  handBasis.makeBasis(handF, handU, handS);
  handDetectedQ.setFromRotationMatrix(handBasis);

  // Поза покоя: пальцы вдоль руки, ладонь вниз, большой палец вперёд
  handF.set(side, 0, 0);
  handS.set(0, 0, -1);
  handU.crossVectors(handS, handF);
  handBasis.makeBasis(handF, handU, handS);
  handRestQ.setFromRotationMatrix(handBasis).invert();

  out.copy(handDetectedQ).multiply(handRestQ);
  return true;
}

// Кисть не может поворачиваться относительно предплечья больше HAND_MAX_BEND
function limitToForearm(lowerBone, qHand) {
  lowerBone.getWorldQuaternion(forearmQ);
  sceneInvQ.copy(currentVrm.scene.quaternion).invert();
  forearmQ.multiply(sceneInvQ); // поворот предплечья относительно позы покоя

  handRelQ.copy(forearmQ).invert().multiply(qHand);
  if (handRelQ.w < 0) {
    handRelQ.set(-handRelQ.x, -handRelQ.y, -handRelQ.z, -handRelQ.w);
  }

  const angle = 2 * Math.acos(Math.min(1, handRelQ.w));
  if (angle > HAND_MAX_BEND) {
    handLimitQ.identity().slerp(handRelQ, HAND_MAX_BEND / angle);
    handRelQ.copy(handLimitQ);
  }

  qHand.copy(forearmQ).multiply(handRelQ);
}

function applyFingers(side, w) {
  const prefix = side === 1 ? 'left' : 'right';

  for (const [finger, mcp, pip, dip, tip] of FINGERS) {
    let p = FINGER_REST;
    let i = FINGER_REST;
    let d = FINGER_REST * 0.7;

    if (w) {
      p = curlAmount(bendAt(w[0], w[mcp], w[pip]), FINGER_MAX.proximal);
      i = curlAmount(bendAt(w[mcp], w[pip], w[dip]), FINGER_MAX.intermediate);
      d = curlAmount(bendAt(w[pip], w[dip], w[tip]), FINGER_MAX.distal);
    }

    setFingerBone(prefix + finger + 'Proximal', 0, -side * p * FINGER_SIGN);
    setFingerBone(prefix + finger + 'Intermediate', 0, -side * i * FINGER_SIGN);
    setFingerBone(prefix + finger + 'Distal', 0, -side * d * FINGER_SIGN);
  }

  // Большой палец: сгибается иначе, поэтому поворот идёт по другой оси (приближённо)
  let t0 = 0;
  let t1 = 0;
  let t2 = 0;
  if (w) {
    t0 = curlAmount(bendAt(w[0], w[1], w[2]), 0.7) * 0.6;
    t1 = curlAmount(bendAt(w[1], w[2], w[3]), 1.0);
    t2 = curlAmount(bendAt(w[2], w[3], w[4]), 1.2);
  }
  setFingerBone(prefix + 'ThumbMetacarpal', side * t0 * THUMB_SIGN, -side * t0 * 0.5 * FINGER_SIGN);
  setFingerBone(prefix + 'ThumbProximal', side * t1 * THUMB_SIGN, -side * t1 * 0.5 * FINGER_SIGN);
  setFingerBone(prefix + 'ThumbDistal', side * t2 * THUMB_SIGN, -side * t2 * 0.5 * FINGER_SIGN);
}

function applyHand(side) {
  const prefix = side === 1 ? 'left' : 'right';
  const data = armsOn ? latestHands[prefix] : null;
  const humanoid = currentVrm.humanoid;
  const handBone = humanoid.getNormalizedBoneNode(prefix + 'Hand');
  const lowerBone = humanoid.getNormalizedBoneNode(prefix + 'LowerArm');

  if (handBone) {
    if (data && WRIST_ON && lowerBone && handWorldQuaternion(side, data.world, worldQ)) {
      limitToForearm(lowerBone, worldQ);
      applyWorldRotation(handBone, worldQ, HAND_SMOOTH);
    } else {
      handBone.quaternion.slerp(identityQ, HAND_SMOOTH);
    }
  }

  applyFingers(side, data ? data.world : null);
}

function applyHands() {
  applyHand(1);
  applyHand(-1);
}

function updateHands() {
  handFrame++;
  if (handFrame % HAND_EVERY !== 0) return;

  const now = performance.now();
  const result = handLandmarker.detectForVideo(video, now);

  const found = result.landmarks.map((image, i) => ({
    image,
    world: result.worldLandmarks[i]
  }));
  // По горизонтали кадра: первая рука слева в кадре (это правая рука человека)
  found.sort((a, b) => a.image[0].x - b.image[0].x);

  let left = null;
  let right = null;

  if (found.length >= 2) {
    const first = found[0];
    const last = found[found.length - 1];
    // Зеркало вкл: рука слева в кадре управляет ЛЕВОЙ рукой модели
    left = mirror ? first : last;
    right = mirror ? last : first;
  } else if (found.length === 1) {
    const hand = found[0];
    let onImageLeft = hand.image[0].x < 0.5;

    // Если видно тело, определяем руку по ближайшему запястью
    if (latestPose) {
      const wristImageLeft = latestPose.image[16];  // правое запястье человека (слева в кадре)
      const wristImageRight = latestPose.image[15]; // левое запястье человека (справа в кадре)
      if ((wristImageLeft.visibility ?? 0) > POSE_VISIBLE && (wristImageRight.visibility ?? 0) > POSE_VISIBLE) {
        const d1 = Math.hypot(hand.image[0].x - wristImageLeft.x, hand.image[0].y - wristImageLeft.y);
        const d2 = Math.hypot(hand.image[0].x - wristImageRight.x, hand.image[0].y - wristImageRight.y);
        onImageLeft = d1 < d2;
      }
    }

    const isModelLeft = mirror ? onImageLeft : !onImageLeft;
    if (isModelLeft) left = hand;
    else right = hand;
  }

  if (left) {
    latestHands.left = { world: smoothLandmarks(left.world, 'hLw', now) };
  } else {
    latestHands.left = null;
    resetFilters('hL');
  }

  if (right) {
    latestHands.right = { world: smoothLandmarks(right.world, 'hRw', now) };
  } else {
    latestHands.right = null;
    resetFilters('hR');
  }
}

function updatePose() {
  poseFrame++;
  if (poseFrame % POSE_EVERY !== 0) return;

  const now = performance.now();
  const result = poseLandmarker.detectForVideo(video, now);
  if (result.landmarks.length > 0 && result.worldLandmarks.length > 0) {
    latestPose = {
      image: smoothLandmarks(result.landmarks[0], 'i', now),
      world: smoothLandmarks(result.worldLandmarks[0], 'w', now)
    };
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
    resetFilters('i');
    resetFilters('w');
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
      '\nзеркало: ' + (mirror ? 'вкл' : 'выкл') +
      '\nтело: ' + (latestPose ? 'видно' : 'не видно') +
      '\nкисти L/R: ' + (latestHands.left ? 'да' : 'нет') + ' / ' + (latestHands.right ? 'да' : 'нет') +
      (armsOn ? '' : ' (руки выключены)') +
      '\nраздельные брови: ' + (browSplit ? 'да' : 'нет') +
      '\nброви L (вверх/вниз): ' + fmt(browDebug.L[0]) + ' / ' + fmt(browDebug.L[1]) +
      '\nброви R (вверх/вниз): ' + fmt(browDebug.R[0]) + ' / ' + fmt(browDebug.R[1]) +
      '\nуправление взглядом: ' + (currentVrm.lookAt ? 'да' : 'нет') +
      '\nвзгляд yaw/pitch: ' + gaze.yaw.toFixed(0) + ' / ' + gaze.pitch.toFixed(0) +
      '\nпики взгляда x+/x-/y+/y-: ' + gazePeaks.xp.toFixed(2) + ' / ' + gazePeaks.xn.toFixed(2) +
      ' / ' + gazePeaks.yp.toFixed(2) + ' / ' + gazePeaks.yn.toFixed(2) +
      '\nкостей торса: ' + torsoBones.length +
      '\nдистанция: ' + (shapes.headZ ?? 0).toFixed(1) + ' см' +
      '\nплечи зазор L/R: ' + fmt(latestShoulders?.left) + ' / ' + fmt(latestShoulders?.right) +
      ' (покой ' + fmt(base.gapLeft) + ' / ' + fmt(base.gapRight) + ')' +
      '\nплечи глубина L/R: ' + fmt(latestShoulders?.depthLeft) + ' / ' + fmt(latestShoulders?.depthRight) +
      '\nкорпус наклон/поворот: ' + fmt(latestTorso?.roll) + ' / ' + fmt(latestTorso?.yaw) +
      ' (покой ' + fmt(base.torsoRoll) + ' / ' + fmt(base.torsoYaw) + ')';
  }

  // Моргание: взгляд вниз тянет веки вниз, это не моргание
  const adjusted = { ...shapes };
  for (const mp of ['Left', 'Right']) {
    const down = Math.max(
      0,
      (shapes['eyeLookDown' + mp] ?? 0) - (neutral['eyeLookDown' + mp] ?? 0)
    );
    adjusted['eyeBlink' + mp] = (shapes['eyeBlink' + mp] ?? 0) - GAZE_BLINK_COMP * down;
  }

  setSmooth('blinkLeft', level(adjusted, 'eyeBlink' + mpSide('Left'), RANGE.blink));
  setSmooth('blinkRight', level(adjusted, 'eyeBlink' + mpSide('Right'), RANGE.blink));
  setSmooth('aa', jawLevel(shapes));
  setSmooth('ou', level(shapes, 'mouthPucker', RANGE.pucker));

  const smile =
    (level(shapes, 'mouthSmileLeft', RANGE.smile) +
      level(shapes, 'mouthSmileRight', RANGE.smile)) / 2;
  setMorphSmooth('Fcl_MTH_Joy', smile);

  applyBrows(shapes);
  updateGazeTarget(shapes, performance.now());

  if (headAngles) {
    updateHeadTarget(shapes);
    updateHeadOffset(shapes);
  }
}

// ---------- Главный цикл ----------
const clock = new THREE.Clock();
let lastVideoTime = -1;

function trackFrame() {
  if (!(faceLandmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime)) return;
  lastVideoTime = video.currentTime;

  // Сначала тело, чтобы плечи и торс были свежими, когда считается лицо
  if (poseLandmarker) {
    updatePose();
  }

  // Кисти считаются только когда включены руки
  if (armsOn && handLandmarker) {
    updateHands();
  }

  const result = faceLandmarker.detectForVideo(video, performance.now());
  if (currentVrm && result.faceBlendshapes.length > 0) {
    lastFaceTime = performance.now();
    applyFace(result);
  }
}

function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();

  // Ошибка трекинга не должна останавливать модель
  try {
    trackFrame();
  } catch (error) {
    if (!trackingErrorShown) {
      console.error('Ошибка трекинга:', error);
      trackingErrorShown = true;
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
    applyHands(); // после рук: кисть крепится к предплечью
    applyGaze(faceLost);

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
