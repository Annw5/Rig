import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { FaceLandmarker, PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

async function openCamera() {
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const modes = [
    { width: { ideal: 1280 }, height: { ideal: 720 } },
    { width: { ideal: 640 }, height: { ideal: 480 } },
    {}
  ];
  let last;
  for (let round = 0; round < 2; round++) {
    // Сначала камера по умолчанию, потом каждая найденная по очереди
    let ids = [undefined];
    try {
      const list = await navigator.mediaDevices.enumerateDevices();
      ids = ids.concat(list.filter((d) => d.kind === 'videoinput' && d.deviceId).map((d) => d.deviceId));
    } catch {}
    for (const id of ids) {
      for (const mode of modes) {
        try {
          return await navigator.mediaDevices.getUserMedia({
            video: id ? { ...mode, deviceId: { exact: id } } : mode
          });
        } catch (e) {
          last = e;
          if (e.name === 'NotAllowedError' || e.name === 'SecurityError') throw e;
        }
      }
    }
    await wait(1500); // даём браузеру освободить камеру и пробуем ещё раз
  }
  throw last;
}

// ======================= НАСТРОЙКИ =======================
// Если что-то двигается наоборот, меняй знаки (1 и -1) здесь

// Голова
const YAW_SIGN = -1, PITCH_SIGN = 1, ROLL_SIGN = -1;
const MAX_ANGLE = { yaw: 1.0, pitch: 0.8, roll: 0.7 };
const MAX_VALID = { yaw: 1.4, pitch: 1.1, roll: 1.0 };
const NECK_SHARE = 0.3;
const HEAD_SMOOTH = 0.4, LOST_SMOOTH = 0.05, LOST_DELAY = 400;

// Положение модели (вперёд, назад, в стороны)
const POS_SIGN = { x: -1, y: 1, z: 1 };
const POS_GAIN = { x: 0.012, y: 0.012, z: 0.03 };
const POS_MAX = { x: 0.6, y: 0.4, z: 1.2 };
const POS_SMOOTH = 0.15;

// Руки
const ARM_SIGN = { x: -1, y: -1, z: -1 };
const ARM_REST_OUT = 0.25, ARM_SMOOTH = 0.3;
const VISIBLE = 0.5, POSE_EVERY = 2;

// Плечи
const SHOULDER_SIGN = 1, SHOULDER_GAIN = 5, SHOULDER_DEAD = 0.01;
const SHOULDER_MAX = 0.7, SHOULDER_SMOOTH = 0.3;

// Корпус
const TORSO_PITCH_SIGN = 1, TORSO_PITCH_GAIN = 0.012, TORSO_PITCH_DEAD = 1.5;
const TORSO_ROLL_SIGN = 1, TORSO_YAW_SIGN = -1, TORSO_DEAD = 0.03;
const TORSO_MAX = { pitch: 0.35, yaw: 0.6, roll: 0.4 };
const TORSO_SMOOTH = 0.2;

// Лицо
const RANGE = { pucker: 0.5, smile: 0.5, brow: 0.5 };
const DEAD = 0.08;
const SORROW_GAIN = 1;
const BLINK_MAX_BASE = 0.25, BLINK_LOOKDOWN = 0.8;
const JAW_MAX_BASE = 0.03;

// Значения тела: они не сохраняются между запусками
const BODY_KEYS = ['headX', 'headY', 'headZ', 'gapLeft', 'gapRight', 'torsoRoll', 'torsoYaw'];

// ======================= ПОМОЩНИКИ =======================
const $ = (id) => document.getElementById(id);
const clamp01 = (v) => Math.min(1, Math.max(0, v));
const limit = (v, max) => Math.max(-max, Math.min(max, v));
const dead = (v, d) => (Math.abs(v) <= d ? 0 : v - Math.sign(v) * d);
const fmt = (v) => (v === undefined ? '-' : v.toFixed(2));
const vis = (p) => (p?.visibility ?? 0) > VISIBLE;

// Шаг к цели: вверх с одной скоростью, вниз с другой
const approach = (current, target, up = 0.5, down = 0.5) => {
  const t = clamp01(target);
  return current + (t - current) * (t > current ? up : down);
};

// Диапазон, который подстраивается под твоё самое сильное движение
function autoRange(minPeak, decay, deadZone) {
  const range = {
    peak: minPeak,
    level(value) {
      const v = Math.max(0, value);
      range.peak = Math.max(v, range.peak * decay, minPeak);
      return clamp01((v - deadZone) / (range.peak - deadZone));
    }
  };
  return range;
}
const jawRange = autoRange(0.08, 0.99, 0.02);
const blinkRanges = { Left: autoRange(0.25, 0.995, 0.05), Right: autoRange(0.25, 0.995, 0.05) };

// ======================= СЦЕНА =======================
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, innerWidth / innerHeight, 0.1, 20);
camera.position.set(0, 0.85, 4.2);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
document.body.appendChild(renderer.domElement);

const sun = new THREE.DirectionalLight(0xffffff, 2);
sun.position.set(1, 1, 1);
scene.add(sun, new THREE.AmbientLight(0xffffff, 0.8));

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// ======================= ИНТЕРФЕЙС =======================
const video = $('video'), cameraButton = $('cameraButton'), calibrateButton = $('calibrateButton');
const resetButton = $('resetButton'), bgButton = $('bgButton'), bodyButton = $('bodyButton');
const debugBox = $('debug'), exprBox = $('exprBox'), panel = $('panel');

addEventListener('keydown', (e) => {
  if (e.code === 'KeyH') panel.hidden = !panel.hidden;
});

// Фон
const BGS = {
  dark: ['тёмный', 0x222222, '#222'],
  green: ['зелёный', 0x00ff00, '#00ff00'],
  transparent: ['прозрачный', null, 'transparent']
};
let bgMode = localStorage.getItem('bgMode');
if (!BGS[bgMode]) bgMode = 'dark';

function applyBackground() {
  const [name, color, css] = BGS[bgMode];
  scene.background = color === null ? null : new THREE.Color(color);
  document.body.style.background = css;
  bgButton.textContent = 'Фон: ' + name;
  localStorage.setItem('bgMode', bgMode);
}
applyBackground();

bgButton.onclick = () => {
  const keys = Object.keys(BGS);
  bgMode = keys[(keys.indexOf(bgMode) + 1) % keys.length];
  applyBackground();
};

// Кнопка рук и плеч
let bodyOn = localStorage.getItem('bodyOn') === '1';
const updateBodyButton = () => {
  bodyButton.textContent = 'Руки и плечи: ' + (bodyOn ? 'вкл' : 'выкл');
};
updateBodyButton();

bodyButton.onclick = async () => {
  bodyOn = !bodyOn;
  localStorage.setItem('bodyOn', bodyOn ? '1' : '0');
  updateBodyButton();
  if (bodyOn) await loadPose();
  else unloadPose();
};

// ======================= ПОКОЙ И КАЛИБРОВКА =======================
let neutral = {};
try {
  neutral = JSON.parse(localStorage.getItem('neutral')) ?? {};
} catch {
  neutral = {};
}
for (const key of BODY_KEYS) delete neutral[key];

function saveNeutral() {
  const saved = { ...neutral };
  for (const key of BODY_KEYS) delete saved[key];
  localStorage.setItem('neutral', JSON.stringify(saved));
}

let calibration = null;
calibrateButton.onclick = () => {
  calibrateButton.textContent = 'Расслабь лицо...';
  setTimeout(() => {
    calibration = { end: performance.now() + 2000, sums: {}, counts: {} };
    calibrateButton.textContent = 'Идёт замер...';
  }, 2000);
};
resetButton.onclick = () => {
  neutral = {};
  localStorage.removeItem('neutral');
};

const level = (shapes, name, range) =>
  Math.max(0, ((shapes[name] ?? 0) - (neutral[name] ?? 0)) / range - DEAD) / (1 - DEAD);
const levelPair = (shapes, name, range) =>
  (level(shapes, name + 'Left', range) + level(shapes, name + 'Right', range)) / 2;

// ======================= МОДЕЛЬ VRM =======================
let vrm = null, faceMeshes = [], torsoBones = [];
const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));

$('fileInput').onchange = (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);

  loader.load(url, (gltf) => {
    if (vrm) {
      scene.remove(vrm.scene);
      VRMUtils.deepDispose(vrm.scene);
    }
    vrm = gltf.userData.vrm;
    VRMUtils.rotateVRM0(vrm);
    scene.add(vrm.scene);
    URL.revokeObjectURL(url);

    exprBox.textContent = 'Выражения модели: ' + Object.keys(vrm.expressionManager.expressionMap).join(', ');

    faceMeshes = [];
    vrm.scene.traverse((o) => {
      if (o.isMesh && o.morphTargetDictionary?.Fcl_MTH_Joy !== undefined) faceMeshes.push(o);
    });
    torsoBones = ['spine', 'chest', 'upperChest']
      .map((n) => vrm.humanoid.getNormalizedBoneNode(n))
      .filter(Boolean);
    console.log('Кусков лица:', faceMeshes.length, 'Костей корпуса:', torsoBones.length, 'VRM', vrm.meta.metaVersion);
  }, undefined, (error) => {
    console.error(error);
    alert('Не удалось загрузить модель. Это точно файл .vrm?');
  });
};

const bone = (name) => vrm.humanoid.getNormalizedBoneNode(name);

function setExpr(name, target, up, down) {
  const e = vrm.expressionManager;
  e.setValue(name, approach(e.getValue(name) ?? 0, target, up, down));
}

// Форма сетки ставится на все куски лица сразу
function setMorph(name, target) {
  let next;
  for (const mesh of faceMeshes) {
    const i = mesh.morphTargetDictionary[name];
    if (i === undefined) continue;
    next ??= approach(mesh.morphTargetInfluences[i], target);
    mesh.morphTargetInfluences[i] = next;
  }
}

// ======================= ПОВОРОТЫ КОСТЕЙ =======================
const euler = new THREE.Euler(), worldQ = new THREE.Quaternion();
const localQ = new THREE.Quaternion(), parentQ = new THREE.Quaternion();

const rot = (x, y, z) => worldQ.setFromEuler(euler.set(x, y, z, 'YXZ'));

// Поворот задан в мире, повороты родителей (позвоночник, плечо...) учитываются сами
function steer(node, qWorld, smooth) {
  if (!node?.parent) return;
  node.parent.getWorldQuaternion(parentQ);
  localQ.copy(parentQ).invert().multiply(qWorld).multiply(vrm.scene.quaternion);
  node.quaternion.slerp(localQ, smooth);
}

// ======================= КАМЕРА И ТРЕКЕРЫ =======================
const MODELS = 'https://storage.googleapis.com/mediapipe-models/';
let faceTracker = null, poseTracker = null, fileset = null, poseLoading = false;
let pose = null, body = null, poseFrame = 0;

// Создаёт трекер на видеокарте, а если не вышло, то на процессоре
async function createTracker(Cls, modelPath, extra) {
  for (const delegate of ['GPU', 'CPU']) {
    try {
      return await Cls.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODELS + modelPath, delegate },
        runningMode: 'VIDEO',
        ...extra
      });
    } catch (error) {
      console.warn('Трекер не запустился на ' + delegate, error);
    }
  }
  return null;
}

async function loadPose() {
  if (!fileset || poseTracker || poseLoading) return;
  poseLoading = true;
  bodyButton.textContent = 'Загрузка тела...';
  poseTracker = await createTracker(
    PoseLandmarker,
    'pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
    { numPoses: 1 }
  );
  poseLoading = false;
  updateBodyButton();
}

function unloadPose() {
  poseTracker?.close();
  poseTracker = null;
  pose = body = null;
}

cameraButton.onclick = async () => {
  try {
    cameraButton.disabled = true;
    cameraButton.textContent = 'Загрузка...';
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720, frameRate: 60 }
    });
    await video.play();

    fileset = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
    );
    faceTracker = await createTracker(
      FaceLandmarker,
      'face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      { numFaces: 1, outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true }
    );
    if (!faceTracker) throw new Error('Не удалось запустить трекинг лица');

    cameraButton.textContent = 'Камера включена';
    if (bodyOn) await loadPose();
  } catch (error) {
    console.error(error);
    alert('Не получилось включить камеру или трекинг. Подробности в консоли (F12).');
    cameraButton.disabled = false;
    cameraButton.textContent = 'Включить камеру';
  }
};

// ======================= ГОЛОВА =======================
const tmpMatrix = new THREE.Matrix4(), tmpPos = new THREE.Vector3();
const tmpQuat = new THREE.Quaternion(), tmpScale = new THREE.Vector3();
const headTarget = { yaw: 0, pitch: 0, roll: 0 };
let lastFaceTime = 0, lastHeadZ;

function readHead(result) {
  const m = result.facialTransformationMatrixes?.[0];
  if (!m) return null;
  tmpMatrix.fromArray(m.data).decompose(tmpPos, tmpQuat, tmpScale);
  euler.setFromQuaternion(tmpQuat, 'YXZ');
  return {
    headYaw: euler.y, headPitch: euler.x, headRoll: euler.z,
    headX: tmpPos.x, headY: tmpPos.y, headZ: tmpPos.z
  };
}

// Считаем цель для головы. Слишком большие углы считаем сбоем и пропускаем
function updateHeadTarget(shapes) {
  const yaw = shapes.headYaw - (neutral.headYaw ?? 0);
  const pitch = shapes.headPitch - (neutral.headPitch ?? 0);
  const roll = shapes.headRoll - (neutral.headRoll ?? 0);
  if (Math.abs(yaw) > MAX_VALID.yaw || Math.abs(pitch) > MAX_VALID.pitch || Math.abs(roll) > MAX_VALID.roll) return;

  headTarget.yaw = limit(yaw * YAW_SIGN, MAX_ANGLE.yaw);
  headTarget.pitch = limit(pitch * PITCH_SIGN, MAX_ANGLE.pitch);
  headTarget.roll = limit(roll * ROLL_SIGN, MAX_ANGLE.roll);
}

function applyHead(smooth) {
  for (const [name, share] of [['neck', NECK_SHARE], ['head', 1]]) {
    steer(bone(name), rot(headTarget.pitch * share, headTarget.yaw * share, headTarget.roll * share), smooth);
  }
}

// Положение всей модели по положению лица
const posTarget = new THREE.Vector3();
function applyBodyPosition(shapes) {
  for (const k of ['headX', 'headY', 'headZ']) neutral[k] ??= shapes[k];
  posTarget.set(
    limit((shapes.headX - neutral.headX) * POS_SIGN.x * POS_GAIN.x, POS_MAX.x),
    limit((shapes.headY - neutral.headY) * POS_SIGN.y * POS_GAIN.y, POS_MAX.y),
    limit((shapes.headZ - neutral.headZ) * POS_SIGN.z * POS_GAIN.z, POS_MAX.z)
  );
  vrm.scene.position.lerp(posTarget, POS_SMOOTH);
}

// ======================= ТЕЛО: ПОЗА, ПЛЕЧИ, КОРПУС, РУКИ =======================
// Зеркало: левая сторона модели повторяет правую сторону человека
const SIDES = [
  { side: 1, shoulder: 'leftShoulder', upper: 'leftUpperArm', lower: 'leftLowerArm', ids: [12, 14, 16], gap: 'gapRight' },
  { side: -1, shoulder: 'rightShoulder', upper: 'rightUpperArm', lower: 'rightLowerArm', ids: [11, 13, 15], gap: 'gapLeft' }
];

// Из точек тела считаем зазор плечо-ухо (для плеч) и углы линии плеч (для корпуса)
function readBody() {
  const p = pose.image, w = pose.world;
  if (!vis(p[11]) || !vis(p[12])) return null;

  const W = video.videoWidth, H = video.videoHeight;
  const width = Math.hypot((p[11].x - p[12].x) * W, (p[11].y - p[12].y) * H);
  if (width < 20) return null;

  const result = {
    torsoRoll: Math.atan2((p[11].y - p[12].y) * H, (p[11].x - p[12].x) * W),
    torsoYaw: Math.atan2(w[11].z - w[12].z, w[11].x - w[12].x)
  };
  if (vis(p[7]) && vis(p[8])) {
    result.gapLeft = ((p[11].y - p[7].y) * H) / width;
    result.gapRight = ((p[12].y - p[8].y) * H) / width;
  }
  return result;
}

function updatePose() {
  if (++poseFrame % POSE_EVERY !== 0) return;
  const r = poseTracker.detectForVideo(video, performance.now());
  if (r.landmarks.length && r.worldLandmarks.length) {
    pose = { image: r.landmarks[0], world: r.worldLandmarks[0] };
    body = readBody();
    if (body) for (const k in body) neutral[k] ??= body[k];
  } else {
    pose = body = null;
  }
}

// Корпус: наклон вперёд и назад от расстояния до камеры, вбок и поворот от линии плеч
const torso = { pitch: 0, yaw: 0, roll: 0 };
function updateTorsoTarget() {
  const dz = lastHeadZ === undefined ? 0 : dead(lastHeadZ - neutral.headZ, TORSO_PITCH_DEAD);
  torso.pitch = limit(dz * TORSO_PITCH_GAIN * TORSO_PITCH_SIGN, TORSO_MAX.pitch);
  torso.roll = torso.yaw = 0;
  if (body) {
    torso.roll = limit(dead(body.torsoRoll - neutral.torsoRoll, TORSO_DEAD) * TORSO_ROLL_SIGN, TORSO_MAX.roll);
    torso.yaw = limit(dead(body.torsoYaw - neutral.torsoYaw, TORSO_DEAD) * TORSO_YAW_SIGN, TORSO_MAX.yaw);
  }
}

function applyTorso() {
  torsoBones.forEach((node, i) => {
    const share = (i + 1) / torsoBones.length;
    steer(node, rot(torso.pitch * share, torso.yaw * share, torso.roll * share), TORSO_SMOOTH);
  });
}

function applyShoulder({ side, shoulder, gap }) {
  let angle = 0;
  if (body && body[gap] !== undefined) {
    angle = limit(dead(neutral[gap] - body[gap], SHOULDER_DEAD) * SHOULDER_GAIN, SHOULDER_MAX);
  }
  steer(bone(shoulder), rot(0, 0, side * angle * SHOULDER_SIGN), SHOULDER_SMOOTH);
}

const armRest = new THREE.Vector3(), armWant = new THREE.Vector3();
const dirUpper = new THREE.Vector3(), dirLower = new THREE.Vector3();
const upperQ = new THREE.Quaternion(), lowerQ = new THREE.Quaternion();

const setDirection = (a, b, out) =>
  out.set((b.x - a.x) * ARM_SIGN.x, (b.y - a.y) * ARM_SIGN.y, (b.z - a.z) * ARM_SIGN.z);

function dirToQuat(side, dir, out) {
  if (dir.lengthSq() < 1e-6) return out.identity();
  return out.setFromUnitVectors(armRest.set(side, 0, 0), armWant.copy(dir).normalize());
}

function applyArm({ side, upper, lower, ids: [s, e, w] }) {
  let upperOk = false, lowerOk = false;
  if (pose) {
    const img = pose.image, wd = pose.world;
    upperOk = vis(img[s]) && vis(img[e]);
    lowerOk = upperOk && vis(img[w]);
    if (upperOk) setDirection(wd[s], wd[e], dirUpper);
    if (lowerOk) setDirection(wd[e], wd[w], dirLower);
  }
  if (!upperOk) dirUpper.set(side * ARM_REST_OUT, -1, 0);   // спокойная поза: руки вниз
  if (!lowerOk) dirLower.copy(dirUpper);

  steer(bone(upper), dirToQuat(side, dirUpper, upperQ), ARM_SMOOTH);
  steer(bone(lower), dirToQuat(side, dirLower, lowerQ), ARM_SMOOTH);
}

// ======================= ПРИМЕНЯЕМ ЛИЦО =======================
let debugCounter = 0;

function applyFace(result) {
  const shapes = {};
  for (const c of result.faceBlendshapes[0].categories) shapes[c.categoryName] = c.score;
  const head = readHead(result);
  if (head) Object.assign(shapes, head);
  if (body) Object.assign(shapes, body);

  // Калибровка: среднее по каждому параметру
  if (calibration) {
    for (const k in shapes) {
      calibration.sums[k] = (calibration.sums[k] ?? 0) + shapes[k];
      calibration.counts[k] = (calibration.counts[k] ?? 0) + 1;
    }
    if (performance.now() > calibration.end) {
      for (const k in calibration.sums) neutral[k] = calibration.sums[k] / calibration.counts[k];
      saveNeutral();
      calibration = null;
      calibrateButton.textContent = 'Калибровка';
    }
  }

  if (++debugCounter % 10 === 0) {
    debugBox.textContent =
      ['eyeBlinkLeft', 'eyeBlinkRight', 'jawOpen', 'mouthSmileLeft', 'mouthSmileRight',
        'browInnerUp', 'browOuterUpLeft', 'browDownLeft']
        .map((n) => n + ': ' + fmt(shapes[n]) + ' (покой ' + fmt(neutral[n]) + ')').join('\n') +
      '\nпик моргания L/R: ' + fmt(blinkRanges.Left.peak) + ' / ' + fmt(blinkRanges.Right.peak) +
      '\nпик рта: ' + fmt(jawRange.peak) +
      '\nдистанция: ' + fmt(shapes.headZ) + ' см' +
      '\nтело: ' + (body ? 'видно' : 'не видно') +
      '\nплечи зазор L/R: ' + fmt(body?.gapLeft) + ' / ' + fmt(body?.gapRight) +
      ' (покой ' + fmt(neutral.gapLeft) + ' / ' + fmt(neutral.gapRight) + ')' +
      '\nкорпус наклон/поворот: ' + fmt(body?.torsoRoll) + ' / ' + fmt(body?.torsoYaw);
  }

  // Моргание: закрывается мгновенно, открывается плавно. Взгляд вниз не считается морганием
  for (const side of ['Left', 'Right']) {
    const blinkBase = Math.min(neutral['eyeBlink' + side] ?? 0, BLINK_MAX_BASE);
    const down = Math.max(0, (shapes['eyeLookDown' + side] ?? 0) - (neutral['eyeLookDown' + side] ?? 0));
    const raw = (shapes['eyeBlink' + side] ?? 0) - blinkBase - BLINK_LOOKDOWN * down;
    setExpr('blink' + side, blinkRanges[side].level(raw), 1, 0.5);
  }

  // Рот и улыбка
  const jawBase = Math.min(neutral.jawOpen ?? 0, JAW_MAX_BASE);
  setExpr('aa', jawRange.level((shapes.jawOpen ?? 0) - jawBase));
  setExpr('ou', level(shapes, 'mouthPucker', RANGE.pucker));
  setMorph('Fcl_MTH_Joy', levelPair(shapes, 'mouthSmile', RANGE.smile));

  // Брови. «Домик» только когда внутренние края поднимаются сильнее внешних
  const outer = levelPair(shapes, 'browOuterUp', RANGE.brow);
  const inner = level(shapes, 'browInnerUp', RANGE.brow);
  setMorph('Fcl_BRW_Surprised', outer);
  setMorph('Fcl_BRW_Angry', levelPair(shapes, 'browDown', RANGE.brow));
  setMorph('Fcl_BRW_Sorrow', (inner - outer) * SORROW_GAIN);

  if (head) {
    updateHeadTarget(shapes);
    applyBodyPosition(shapes);
    lastHeadZ = shapes.headZ;
  }
}

// ======================= ГЛАВНЫЙ ЦИКЛ =======================
const clock = new THREE.Clock();
let lastVideoTime = -1, errorShown = false;

function track() {
  if (!(faceTracker && video.readyState >= 2 && video.currentTime !== lastVideoTime)) return;
  lastVideoTime = video.currentTime;

  if (bodyOn && poseTracker) updatePose();   // тело первым: плечи и корпус свежие
  const result = faceTracker.detectForVideo(video, performance.now());
  if (vrm && result.faceBlendshapes.length) {
    lastFaceTime = performance.now();
    applyFace(result);
  }
}

function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();

  try {
    track();
  } catch (e) {   // ошибка трекинга не должна останавливать модель
    if (!errorShown) {
      console.error('Ошибка трекинга:', e);
      errorShown = true;
    }
  }

  if (vrm) {
    const faceLost = faceTracker && performance.now() - lastFaceTime > LOST_DELAY;
    if (faceLost) {
      headTarget.yaw = headTarget.pitch = headTarget.roll = 0;
      lastHeadZ = undefined;
    }

    // Порядок важен: сначала туловище, потом то, что к нему прикреплено
    updateTorsoTarget();
    applyTorso();
    SIDES.forEach(applyShoulder);
    applyHead(faceLost ? LOST_SMOOTH : HEAD_SMOOTH);
    SIDES.forEach(applyArm);

    vrm.update(delta);
  }
  renderer.render(scene, camera);
}
animate();
