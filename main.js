import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { FaceLandmarker, PoseLandmarker, HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// ======================= НАСТРОЙКИ =======================
// Если что-то двигается наоборот, меняй знаки (1 и -1) здесь
const FILTER = { min: 1, beta: 1 };   // min меньше = меньше дрожь, но больше задержка

const YAW_SIGN = -1, PITCH_SIGN = 1, ROLL_SIGN = -1;
const MAX_ANGLE = { yaw: 1.0, pitch: 0.8, roll: 0.7 };
const MAX_VALID = { yaw: 1.4, pitch: 1.1, roll: 1.0 };
const NECK_SHARE = 0.3, HEAD_SMOOTH = 0.4, LOST_SMOOTH = 0.05, LOST_DELAY = 400;

const GAZE = { yaw: -100, pitch: 80 };   // взгляд. Плохо реагирует: увеличь. Идёт не туда: смени знак

const POS_SIGN = { x: -1, y: 1, z: 1 };
const POS_GAIN = { x: 0.012, y: 0.012, z: 0.03 };
const POS_MAX = { x: 0.6, y: 0.4, z: 1.2 };
const POS_SMOOTH = 0.15;

const ARM_SIGN = { x: -1, y: -1, z: -1 };
const ARM_REST_OUT = 0.25, ARM_SMOOTH = 0.3, VISIBLE = 0.5, POSE_EVERY = 2;
const FINGER_SMOOTH = 0.5, FINGER_MAX = 1.6;
// Большой палец гнётся вокруг Y (к ладони) и немного вокруг Z. Гнётся не туда: смени знак
const THUMB = { y: 1, z: 0.4 };

const SHOULDER_SIGN = 1, SHOULDER_GAIN = 5, SHOULDER_DEAD = 0.01, SHOULDER_MAX = 0.7, SHOULDER_SMOOTH = 0.3;

const TORSO_PITCH_SIGN = 1, TORSO_PITCH_GAIN = 0.012, TORSO_PITCH_DEAD = 1.5;
const TORSO_ROLL_SIGN = 1, TORSO_YAW_SIGN = -1, TORSO_DEAD = 0.03;
const TORSO_MAX = { pitch: 0.35, yaw: 0.6, roll: 0.4 };
const TORSO_SMOOTH = 0.2;

const RANGE = { pucker: 0.5, smile: 0.5, brow: 0.5 };
const DEAD = 0.08, SORROW_GAIN = 1, BLINK_MAX_BASE = 0.25, BLINK_LOOKDOWN = 0.8, JAW_MAX_BASE = 0.03;

// Пальцы: название, номер первой точки руки, названия суставов
const P = ['Proximal', 'Intermediate', 'Distal'];
const FINGERS = [['Thumb', 1, ['Metacarpal', 'Proximal', 'Distal']], ['Index', 5, P], ['Middle', 9, P], ['Ring', 13, P], ['Little', 17, P]];

// Значения тела не сохраняются между запусками
const BODY_KEYS = ['headX', 'headY', 'headZ', 'gapLeft', 'gapRight', 'torsoRoll', 'torsoYaw'];

// ======================= ПОМОЩНИКИ =======================
const $ = (id) => document.getElementById(id);
const clamp01 = (v) => Math.min(1, Math.max(0, v));
const limit = (v, max) => Math.max(-max, Math.min(max, v));
const dead = (v, d) => (Math.abs(v) <= d ? 0 : v - Math.sign(v) * d);
const fmt = (v) => (v === undefined ? '-' : v.toFixed(2));
const vis = (p) => (p?.visibility ?? 0) > VISIBLE;
const other = (s) => (s === 'Left' ? 'Right' : 'Left');

// Шаг к цели: вверх с одной скоростью, вниз с другой
const approach = (cur, target, up = 0.5, down = 0.5) => {
  const t = clamp01(target);
  return cur + (t - cur) * (t > cur ? up : down);
};

// Диапазон, который подстраивается под твоё самое сильное движение
function autoRange(minPeak, decay, deadZone) {
  const r = {
    peak: minPeak,
    level(value) {
      const v = Math.max(0, value);
      r.peak = Math.max(v, r.peak * decay, minPeak);
      return clamp01((v - deadZone) / (r.peak - deadZone));
    }
  };
  return r;
}
const jawRange = autoRange(0.08, 0.99, 0.02);
const blinkRanges = { Left: autoRange(0.25, 0.995, 0.05), Right: autoRange(0.25, 0.995, 0.05) };

// Фильтр One Euro: гасит дрожь в покое, не отставая при быстром движении
function oneEuro() {
  let x, dx = 0, t0;
  const alpha = (cutoff, dt) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));
  return (v, t) => {
    if (x === undefined) { x = v; t0 = t; return v; }
    const dt = Math.max(t - t0, 1e-3);
    t0 = t;
    dx += alpha(1, dt) * ((v - x) / dt - dx);
    x += alpha(FILTER.min + FILTER.beta * Math.abs(dx), dt) * (v - x);
    return x;
  };
}
const filters = {};
const smooth = (key, v, t) => (filters[key] ??= oneEuro())(v, t);

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

    faceMeshes = [];
    vrm.scene.traverse((o) => {
      if (o.isMesh && o.morphTargetDictionary?.Fcl_MTH_Joy !== undefined) faceMeshes.push(o);
    });
    torsoBones = ['spine', 'chest', 'upperChest'].map(bone).filter(Boolean);

    // Диагностика модели
    const fingers = ['left', 'right']
      .flatMap((s) => FINGERS.flatMap(([f, , parts]) => parts.map((p) => s + f + p)))
      .filter(bone).length;
    exprBox.textContent =
      'Кости пальцев: ' + fingers + ' из 30\nВзгляд: ' + (vrm.lookAt ? 'есть' : 'нет в модели') +
      '\nVRM ' + vrm.meta.metaVersion + ', кусков лица: ' + faceMeshes.length;
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

// Если в модели есть формы с _L и _R, левая и правая стороны двигаются отдельно, иначе вместе
function pairMorph(name, left, right) {
  if (faceMeshes[0]?.morphTargetDictionary[name + '_L'] !== undefined) {
    setMorph(name + '_L', right);   // зеркало: левая сторона модели повторяет правую сторону человека
    setMorph(name + '_R', left);
  } else {
    setMorph(name, (left + right) / 2);
  }
}

// ======================= ПОВОРОТЫ КОСТЕЙ =======================
const euler = new THREE.Euler(), worldQ = new THREE.Quaternion();
const localQ = new THREE.Quaternion(), parentQ = new THREE.Quaternion();
const rot = (x, y, z) => worldQ.setFromEuler(euler.set(x, y, z, 'YXZ'));

// Поворот задан в мире, повороты родителей учитываются сами
function steer(node, qWorld, smoothing) {
  if (!node?.parent) return;
  node.parent.getWorldQuaternion(parentQ);
  localQ.copy(parentQ).invert().multiply(qWorld).multiply(vrm.scene.quaternion);
  node.quaternion.slerp(localQ, smoothing);
}

// ======================= КАМЕРА И ТРЕКЕРЫ =======================
const MODELS = 'https://storage.googleapis.com/mediapipe-models/';
let faceTracker = null, poseTracker = null, handTracker = null, fileset = null, poseLoading = false;
let pose = null, body = null, hands = {}, poseFrame = 0, lastTrackerError = null;

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
      lastTrackerError = error;
      console.warn('Трекер не запустился на ' + delegate, error);
    }
  }
  return null;
}

async function loadPose() {
  if (!fileset || poseTracker || poseLoading) return;
  poseLoading = true;
  bodyButton.textContent = 'Загрузка тела...';
  poseTracker = await createTracker(PoseLandmarker, 'pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task', { numPoses: 1 });
  handTracker = await createTracker(HandLandmarker, 'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    { numHands: 2, minHandDetectionConfidence: 0.3, minHandPresenceConfidence: 0.3, minTrackingConfidence: 0.3 });
  poseLoading = false;
  updateBodyButton();
}

function unloadPose() {
  poseTracker?.close();
  handTracker?.close();
  poseTracker = handTracker = pose = body = null;
  hands = {};
}

function stopCamera() {
  video.srcObject?.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
}

// Пробует режимы и все найденные камеры, при неудаче ждёт и повторяет
async function openCamera() {
  const modes = [
    { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60 } },
    { width: { ideal: 640 }, height: { ideal: 480 } },
    {}
  ];
  let lastError;
  for (let round = 0; round < 2; round++) {
    let ids = [undefined];
    try {
      const list = await navigator.mediaDevices.enumerateDevices();
      ids = ids.concat(list.filter((d) => d.kind === 'videoinput' && d.deviceId).map((d) => d.deviceId));
    } catch {}
    for (const id of ids) {
      for (const mode of modes) {
        try {
          return await navigator.mediaDevices.getUserMedia({ video: id ? { ...mode, deviceId: { exact: id } } : mode });
        } catch (error) {
          lastError = error;
          if (error.name === 'NotAllowedError' || error.name === 'SecurityError') throw error;
        }
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
  throw lastError;
}

const BUSY = 'Камеру, скорее всего, занимает другое приложение или вкладка (OBS, Zoom, Discord). Закрой их, перезагрузи страницу и нажми ещё раз.';
const DENIED = 'Нет доступа к камере. Разреши её для этого сайта и перезагрузи страницу.';
const NONE = 'Камера не найдена. Проверь, что она подключена.';
const CAMERA_ERRORS = {
  NotReadableError: BUSY, AbortError: BUSY, NotAllowedError: DENIED,
  SecurityError: DENIED, NotFoundError: NONE, OverconstrainedError: NONE
};

cameraButton.onclick = async () => {
  let stage = 'включение камеры';
  try {
    cameraButton.disabled = true;
    cameraButton.textContent = 'Загрузка...';
    stopCamera();
    video.srcObject = await openCamera();

    stage = 'запуск видео';
    await video.play();

    stage = 'загрузка файлов трекинга';
    fileset = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');

    stage = 'запуск трекера лица';
    faceTracker = await createTracker(
      FaceLandmarker,
      'face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      { numFaces: 1, outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true }
    );
    if (!faceTracker) throw new Error('трекер лица не запустился: ' + lastTrackerError?.message);

    cameraButton.textContent = 'Камера включена';
    stage = 'запуск трекера тела';
    if (bodyOn) await loadPose();
  } catch (error) {
    console.error('Этап: ' + stage, error);
    stopCamera();
    cameraButton.disabled = false;
    cameraButton.textContent = 'Включить камеру';
    alert((CAMERA_ERRORS[error?.name] ?? 'Не получилось включить камеру или трекинг.') +
      '\n\nЭтап: ' + stage + '\nПричина: ' + (error?.name ?? '') + ' ' + (error?.message ?? ''));
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
  return { headYaw: euler.y, headPitch: euler.x, headRoll: euler.z, headX: tmpPos.x, headY: tmpPos.y, headZ: tmpPos.z };
}

// Слишком большие углы считаем сбоем и пропускаем
function updateHeadTarget(shapes) {
  const yaw = shapes.headYaw - (neutral.headYaw ?? 0);
  const pitch = shapes.headPitch - (neutral.headPitch ?? 0);
  const roll = shapes.headRoll - (neutral.headRoll ?? 0);
  if (Math.abs(yaw) > MAX_VALID.yaw || Math.abs(pitch) > MAX_VALID.pitch || Math.abs(roll) > MAX_VALID.roll) return;
  headTarget.yaw = limit(yaw * YAW_SIGN, MAX_ANGLE.yaw);
  headTarget.pitch = limit(pitch * PITCH_SIGN, MAX_ANGLE.pitch);
  headTarget.roll = limit(roll * ROLL_SIGN, MAX_ANGLE.roll);
}

function applyHead(smoothing) {
  for (const [name, share] of [['neck', NECK_SHARE], ['head', 1]]) {
    steer(bone(name), rot(headTarget.pitch * share, headTarget.yaw * share, headTarget.roll * share), smoothing);
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

// ======================= ТЕЛО: ПОЗА, ПЛЕЧИ, КОРПУС, РУКИ, ПАЛЬЦЫ =======================
// Зеркало: левая сторона модели повторяет правую сторону человека
const SIDES = [
  { side: 1, shoulder: 'leftShoulder', upper: 'leftUpperArm', lower: 'leftLowerArm', ids: [12, 14, 16], gap: 'gapRight' },
  { side: -1, shoulder: 'rightShoulder', upper: 'rightUpperArm', lower: 'rightLowerArm', ids: [11, 13, 15], gap: 'gapLeft' }
];

// Зазор плечо-ухо (для плеч) и углы линии плеч (для корпуса)
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
  const now = performance.now(), t = now / 1000;
  const r = poseTracker?.detectForVideo(video, now);
  const h = handTracker?.detectForVideo(video, now);

  hands = {};
  h?.worldLandmarks.forEach((lm, i) => { hands[h.handedness[i][0].categoryName] = lm; });

  if (r?.landmarks.length && r.worldLandmarks.length) {
    pose = { image: r.landmarks[0], world: r.worldLandmarks[0] };
    for (const i of [11, 12, 13, 14, 15, 16]) for (const a of 'xyz') pose.world[i][a] = smooth(i + a, pose.world[i][a], t);
    body = readBody();
    if (body) for (const k in body) neutral[k] ??= (body[k] = smooth(k, body[k], t));
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

// Пальцы: сгиб каждого сустава равен углу между соседними отрезками пальца

const va = new THREE.Vector3(), vb = new THREE.Vector3();
const seg = (lm, p, k, v) => v.set(lm[p[k + 1]].x - lm[p[k]].x, lm[p[k + 1]].y - lm[p[k]].y, lm[p[k + 1]].z - lm[p[k]].z);

let fist = false, fingerCurl;
addEventListener('keydown', (e) => { if (e.code === 'KeyF') fist = !fist; });

function applyHand([name, side]) {
  const lm = hands[name], flip = vrm.meta.metaVersion === '0' ? -1 : 1;
  for (const [finger, start, parts] of FINGERS) {
    const p = [0, start, start + 1, start + 2, start + 3];
    parts.forEach((part, j) => {
      const curl = fist ? 1.2 : lm ? Math.min(seg(lm, p, j, va).angleTo(seg(lm, p, j + 1, vb)), FINGER_MAX) : 0.15;
      if (finger === 'Index' && j === 0 && (lm || fist)) fingerCurl = curl;
      const q = finger === 'Thumb'
        ? rot(0, side * THUMB.y * curl, -side * flip * THUMB.z * curl)
        : rot(0, 0, -side * flip * curl);
      bone(name.toLowerCase() + finger + part)?.quaternion.slerp(q, FINGER_SMOOTH);
    });
  }
}

// ======================= ПРИМЕНЯЕМ ЛИЦО =======================
let debugCounter = 0;

function applyFace(result) {
  const shapes = {}, t = performance.now() / 1000;
  for (const c of result.faceBlendshapes[0].categories) shapes[c.categoryName] = c.score;
  const head = readHead(result);
  if (head) {
    for (const k in head) head[k] = smooth(k, head[k], t);
    Object.assign(shapes, head);
  }
  if (body) Object.assign(shapes, body);

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
      ['eyeBlinkLeft', 'eyeBlinkRight', 'jawOpen', 'mouthSmileLeft', 'mouthSmileRight', 'browInnerUp', 'browOuterUpLeft', 'browDownLeft']
        .map((n) => n + ': ' + fmt(shapes[n]) + ' (покой ' + fmt(neutral[n]) + ')').join('\n') +
      '\nпик моргания L/R: ' + fmt(blinkRanges.Left.peak) + ' / ' + fmt(blinkRanges.Right.peak) +
      '\nпик рта: ' + fmt(jawRange.peak) +
      '\nдистанция: ' + fmt(shapes.headZ) + ' см' +
      '\nтело: ' + (body ? 'видно' : 'не видно') +
      '\nкисти: ' + (handTracker ? Object.keys(hands).join(' ') || 'не вижу' : 'трекер не запущен') +
      '\nсгиб указательного: ' + fmt(fingerCurl) + (fist ? ' (тест F)' : '') +
      '\nплечи зазор L/R: ' + fmt(body?.gapLeft) + ' / ' + fmt(body?.gapRight) +
      ' (покой ' + fmt(neutral.gapLeft) + ' / ' + fmt(neutral.gapRight) + ')' +
      '\nкорпус наклон/поворот: ' + fmt(body?.torsoRoll) + ' / ' + fmt(body?.torsoYaw);
  }

  // Моргание: закрывается мгновенно, открывается плавно. Взгляд вниз не считается морганием
  for (const side of ['Left', 'Right']) {
    const base = Math.min(neutral['eyeBlink' + side] ?? 0, BLINK_MAX_BASE);
    const down = Math.max(0, (shapes['eyeLookDown' + side] ?? 0) - (neutral['eyeLookDown' + side] ?? 0));
    const raw = (shapes['eyeBlink' + side] ?? 0) - base - BLINK_LOOKDOWN * down;
    setExpr('blink' + other(side), blinkRanges[side].level(raw), 1, 0.5);
  }

  // Зрачки: поворот взгляда по числам трекера
  if (vrm.lookAt) {
    const g = (n) => (shapes[n] ?? 0) - (neutral[n] ?? 0);
    const yaw = (g('eyeLookOutLeft') - g('eyeLookInLeft') + g('eyeLookInRight') - g('eyeLookOutRight')) / 2;
    const pitch = (g('eyeLookUpLeft') + g('eyeLookUpRight') - g('eyeLookDownLeft') - g('eyeLookDownRight')) / 2;
    vrm.lookAt.yaw = smooth('gazeYaw', yaw * GAZE.yaw, t);
    vrm.lookAt.pitch = smooth('gazePitch', pitch * GAZE.pitch, t);
  }

  // Рот и улыбка
  setExpr('aa', jawRange.level((shapes.jawOpen ?? 0) - Math.min(neutral.jawOpen ?? 0, JAW_MAX_BASE)));
  setExpr('ou', level(shapes, 'mouthPucker', RANGE.pucker));
  pairMorph('Fcl_MTH_Joy', level(shapes, 'mouthSmileLeft', RANGE.smile), level(shapes, 'mouthSmileRight', RANGE.smile));

  // Брови. «Домик» только когда внутренние края поднимаются сильнее внешних
  const inner = level(shapes, 'browInnerUp', RANGE.brow);
  const brow = (side) => {
    const up = level(shapes, 'browOuterUp' + side, RANGE.brow);
    return { Surprised: up, Angry: level(shapes, 'browDown' + side, RANGE.brow), Sorrow: (inner - up) * SORROW_GAIN };
  };
  const bl = brow('Left'), br = brow('Right');
  for (const k in bl) pairMorph('Fcl_BRW_' + k, bl[k], br[k]);

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

  if (bodyOn && (poseTracker || handTracker)) updatePose();   // тело первым: плечи и корпус свежие
  const result = faceTracker.detectForVideo(video, performance.now());
  if (vrm && result.faceBlendshapes.length) {
    lastFaceTime = performance.now();
    applyFace(result);
  }
}

function animate() {
  requestAnimationFrame(animate);
  try {
    track();
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
      [['Left', 1], ['Right', -1]].forEach(applyHand);
    }
  } catch (e) {   // ошибка не должна замораживать модель
    if (!errorShown) {
      console.error('Ошибка:', e);
      errorShown = true;
    }
  }
  vrm?.update(clock.getDelta());
  renderer.render(scene, camera);
}
animate();
