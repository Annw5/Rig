import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { FaceLandmarker, PoseLandmarker, HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// ======================= НАСТРОЙКИ =======================
// Если что-то работает наоборот, меняй знаки (1 / -1) здесь
const YAW_SIGN = -1, PITCH_SIGN = 1, ROLL_SIGN = -1;      // голова
const MAX_ANGLE = { yaw: 1.0, pitch: 0.8, roll: 0.7 };
const NECK_SHARE = 0.3;
const MAX_VALID = { yaw: 1.4, pitch: 1.1, roll: 1.0 };
const LOST_DELAY = 400;

const MAX_STEP = 0.35;        // максимум поворота кости за кадр, радианы (защита от «отлёта»)
const HEAD_SMOOTH = 0.4, ARM_SMOOTH = 0.2, SHOULDER_SMOOTH = 0.2, TORSO_SMOOTH = 0.15, HAND_SMOOTH = 0.3, FINGER_SMOOTH = 0.4;

const ARM_SIGN = { x: -1, y: -1, z: -1 };
const ARM_REST_OUT = 0.25;
const VIS = 0.5;
const POSE_EVERY = 2, HAND_EVERY = 3;

const SHOULDER_SIGN = 1, SHOULDER_GAIN = 3, SHOULDER_DEAD = 0.03, SHOULDER_MAX = 0.5;
const DEPTH_SIGN = -1, DEPTH_GAIN = 3, DEPTH_DEAD = 0.02, DEPTH_MAX = 0.3;

const TORSO_ROLL_SIGN = 1, TORSO_YAW_SIGN = -1, TORSO_DEAD = 0.05;
const TORSO_MAX = { yaw: 0.8, roll: 0.6 };

const HAND_FLIP = { left: true, right: true };
const HAND_MAX_BEND = 1.7;
const FINGER_SIGN = 1, THUMB_SIGN = 1, FINGER_GAIN = 1.1, FINGER_DEAD = 0.08, FINGER_REST = 0.12;
const FINGER_MAX = { proximal: 1.5, intermediate: 1.9, distal: 1.3 };

const FILTER_MIN_CUTOFF = 0.5, FILTER_BETA = 5;

const GAZE_YAW_SIGN = 1, GAZE_PITCH_SIGN = 1;
const GAZE_DEAD = { x: 0.03, y: 0.02 };
const GAZE_MIN_PEAK = { x: 0.2, y: 0.1 }, GAZE_MAX_PEAK = { x: 0.8, y: 0.6 };
const GAZE_PEAK_DECAY = 0.9995, GAZE_SMOOTH = 0.5, GAZE_FILTER_CUTOFF = 1.5, GAZE_MAX = 90;
const GAZE_BLINK_COMP = 0.8;
const BLINK_HOLD = 0.25;

const RANGE = { blink: 0.5, pucker: 0.5, smile: 0.5, brow: 0.5 };
const DEAD = 0.08;
const BROW_UP = 1.3, BROW_DOWN = 1.8, BROW_ASYM = 0.9, BROW_ASYM_DEAD = 0.2, SORROW_GAIN = 0.7;
const BROW_MORPHS = ['Fcl_BRW_Surprised', 'Fcl_BRW_Angry', 'Fcl_BRW_Sorrow'];
const BROW_SEAM = 0.015;
const BROW_SWAP = false;
const FACE_SWAP = false;

const BODY_KEYS = ['gapLeft', 'gapRight', 'depthLeft', 'depthRight', 'torsoRoll', 'torsoYaw'];

// ======================= ПОМОЩНИКИ =======================
const $ = (id) => document.getElementById(id);
const clamp01 = (v) => Math.min(1, Math.max(0, v));
const limit = (v, m) => Math.max(-m, Math.min(m, v));
const dead = (v, d) => (Math.abs(v) <= d ? 0 : v - Math.sign(v) * d);
const fmt = (v) => (v === undefined ? '-' : v.toFixed(2));
const vis = (p) => (p?.visibility ?? 0) > VIS;

// Фильтр One Euro: гасит дрожание в покое и почти не тормозит быстрые движения
const alpha = (cutoff, dt) => 1 / (1 + 1 / (2 * Math.PI * cutoff) / dt);
class OneEuro {
  constructor(minCutoff) { this.minCutoff = minCutoff; this.x = null; this.dx = 0; this.t = 0; }
  filter(v, t) {
    if (this.x === null) { this.x = v; this.t = t; return v; }
    const dt = Math.max((t - this.t) / 1000, 0.001);
    this.dx += alpha(1, dt) * ((v - this.x) / dt - this.dx);
    this.x += alpha(this.minCutoff + FILTER_BETA * Math.abs(this.dx), dt) * (v - this.x);
    this.t = t;
    return this.x;
  }
}
const filters = {};
const filt = (key, v, t, cutoff = FILTER_MIN_CUTOFF) => (filters[key] ??= new OneEuro(cutoff)).filter(v, t);
const resetFilters = (prefix) => { for (const k in filters) if (k.startsWith(prefix)) delete filters[k]; };
const smoothPts = (list, kind, t) => list.map((p, i) => ({
  x: filt(kind + i + 'x', p.x, t), y: filt(kind + i + 'y', p.y, t), z: filt(kind + i + 'z', p.z, t),
  visibility: p.visibility
}));

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
const resetButton = $('resetButton'), bgButton = $('bgButton'), armsButton = $('bodyButton');
const debugBox = $('debug'), exprBox = $('exprBox'), panel = $('panel');
const mirrorButton = document.createElement('button');
armsButton.insertAdjacentElement('afterend', mirrorButton);

addEventListener('keydown', (e) => { if (e.code === 'KeyH') panel.hidden = !panel.hidden; });

// Фон
const BGS = { dark: ['тёмный', 0x222222, '#222'], green: ['зелёный', 0x00ff00, '#00ff00'], transparent: ['прозрачный', null, 'transparent'] };
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

// Зеркало: вкл = модель повторяет твоё отражение, выкл = правая рука управляет правой
let mirror = localStorage.getItem('mirror') !== '0';
const mir = () => (mirror ? 1 : -1);
function updateMirrorButton() {
  mirrorButton.textContent = 'Зеркало: ' + (mirror ? 'вкл' : 'выкл');
  video.style.transform = mirror ? 'scaleX(-1)' : 'none';
}
updateMirrorButton();
mirrorButton.onclick = () => {
  mirror = !mirror;
  localStorage.setItem('mirror', mirror ? '1' : '0');
  updateMirrorButton();
  hands.left = hands.right = null;
  resetFilters('h');
};
function mpSide(modelSide, extraSwap = false) {
  const flip = (!mirror !== FACE_SWAP) !== extraSwap;
  return flip ? (modelSide === 'Left' ? 'Right' : 'Left') : modelSide;
}

// Руки (кнопка управляет только руками, кистями и пальцами)
let armsOn = localStorage.getItem('armsOn') === '1';
const updateArmsButton = () => { armsButton.textContent = 'Руки: ' + (armsOn ? 'вкл' : 'выкл'); };
updateArmsButton();
armsButton.onclick = async () => {
  armsOn = !armsOn;
  localStorage.setItem('armsOn', armsOn ? '1' : '0');
  updateArmsButton();
  if (armsOn) await loadHands();
  else unloadHands();
};

// ======================= НАСТРОЙКИ ПОКОЯ =======================
let neutral = {};
try { neutral = JSON.parse(localStorage.getItem('neutral')) ?? {}; } catch { neutral = {}; }
let base = {};
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
  base = {};
  localStorage.removeItem('neutral');
  peaks.xp = peaks.xn = GAZE_MIN_PEAK.x;
  peaks.yp = peaks.yn = GAZE_MIN_PEAK.y;
};

// ======================= МОДЕЛЬ VRM =======================
let vrm = null, faceMeshes = [], torsoBones = [], browSplit = false;
const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));
const vrmFlip = () => (vrm?.meta.metaVersion === '0' ? -1 : 1);

function maskMorph(src, position, side) {
  const out = new Float32Array(src.count * 3);
  for (let i = 0; i < src.count; i++) {
    const w = clamp01((position.getX(i) * side + BROW_SEAM) / (2 * BROW_SEAM));
    out[i * 3] = src.getX(i) * w;
    out[i * 3 + 1] = src.getY(i) * w;
    out[i * 3 + 2] = src.getZ(i) * w;
  }
  return new THREE.BufferAttribute(out, 3);
}

function splitBrowMorphs() {
  browSplit = false;
  const leftSign = vrm.meta.metaVersion === '0' ? -1 : 1;
  const done = new Map();
  for (const mesh of faceMeshes) {
    const geo = mesh.geometry, pos = geo.morphAttributes.position, nor = geo.morphAttributes.normal;
    if (!pos || !geo.attributes.position) continue;
    if (!done.has(geo)) {
      const added = {};
      for (const name of BROW_MORPHS) {
        const src = mesh.morphTargetDictionary[name];
        if (src === undefined) continue;
        for (const [sfx, side] of [['_L', leftSign], ['_R', -leftSign]]) {
          added[name + sfx] = pos.length;
          pos.push(maskMorph(pos[src], geo.attributes.position, side));
          if (nor?.[src]) nor.push(maskMorph(nor[src], geo.attributes.position, side));
        }
      }
      done.set(geo, added);
    }
    const added = done.get(geo);
    for (const key in added) { mesh.morphTargetDictionary[key] = added[key]; browSplit = true; }
    while (mesh.morphTargetInfluences.length < pos.length) mesh.morphTargetInfluences.push(0);
  }
}

$('fileInput').onchange = (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  loader.load(url, (gltf) => {
    if (vrm) { scene.remove(vrm.scene); VRMUtils.deepDispose(vrm.scene); }
    vrm = gltf.userData.vrm;
    VRMUtils.rotateVRM0(vrm);
    scene.add(vrm.scene);
    URL.revokeObjectURL(url);

    const names = Object.keys(vrm.expressionManager.expressionMap);
    exprBox.textContent = 'Выражения модели (' + names.length + '):\n' + names.join(', ');

    faceMeshes = [];
    vrm.scene.traverse((o) => {
      const d = o.isMesh && o.morphTargetDictionary;
      if (d && (d.Fcl_MTH_Joy !== undefined || BROW_MORPHS.some((n) => d[n] !== undefined))) faceMeshes.push(o);
    });
    torsoBones = ['spine', 'chest', 'upperChest'].map((n) => vrm.humanoid.getNormalizedBoneNode(n)).filter(Boolean);
    splitBrowMorphs();
    gaze.yaw = gaze.pitch = 0;
    console.log('Лицо:', faceMeshes.length, 'Торс:', torsoBones.length, 'Брови раздельно:', browSplit,
      'lookAt:', !!vrm.lookAt, 'VRM', vrm.meta.metaVersion);
  }, undefined, (err) => {
    console.error(err);
    alert('Не удалось загрузить модель. Это точно файл .vrm?');
  });
};

// ======================= КАМЕРА И ТРЕКЕРЫ =======================
let faceT = null, poseT = null, handT = null, fileset = null;
let pose = null, body = null;
const hands = { left: null, right: null };

async function createTracker(Cls, modelPath, extra) {
  for (const delegate of ['GPU', 'CPU']) {
    try {
      return await Cls.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: modelPath, delegate }, runningMode: 'VIDEO', ...extra
      });
    } catch (e) { console.warn('Трекер не запустился на ' + delegate, e); }
  }
  return null;
}
const MODELS = 'https://storage.googleapis.com/mediapipe-models/';
const loadPose = async () => {
  poseT ??= await createTracker(PoseLandmarker, MODELS + 'pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task', { numPoses: 1 });
};
async function loadHands() {
  if (!fileset || handT) return;
  armsButton.textContent = 'Загрузка кистей...';
  handT = await createTracker(HandLandmarker, MODELS + 'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', { numHands: 2 });
  updateArmsButton();
}
function unloadHands() {
  handT?.close();
  handT = null;
  hands.left = hands.right = null;
  resetFilters('h');
}

function stopCamera() {
  video.srcObject?.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
}

// Камера: перебирает режимы и все найденные камеры, при неудаче ждёт и повторяет
async function openCamera() {
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const modes = [
    { width: { ideal: 1280 }, height: { ideal: 720 } },
    { width: { ideal: 640 }, height: { ideal: 480 } },
    {}
  ];
  let last;
  for (let round = 0; round < 2; round++) {
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
    await wait(1500);
  }
  throw last;
}

const CAMERA_ERRORS = {
  NotReadableError: 'Камеру, скорее всего, занимает другое приложение или вкладка (OBS, Zoom, Discord, Telegram). Закрой их, перезагрузи страницу и нажми кнопку ещё раз.',
  AbortError: 'Камеру, скорее всего, занимает другое приложение или вкладка. Закрой их и нажми кнопку ещё раз.',
  NotAllowedError: 'Нет доступа к камере. Разреши её для этого сайта (значок замка у адреса) и перезагрузи страницу.',
  SecurityError: 'Нет доступа к камере. Разреши её для этого сайта и перезагрузи страницу.',
  NotFoundError: 'Камера не найдена. Проверь, что она подключена.',
  OverconstrainedError: 'Камера не найдена. Проверь, что она подключена.'
};

cameraButton.onclick = async () => {
  try {
    cameraButton.disabled = true;
    cameraButton.textContent = 'Загрузка...';
    stopCamera();
    video.srcObject = await openCamera();   // раньше здесь был прямой getUserMedia
    await video.play();
    fileset = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
    faceT = await createTracker(FaceLandmarker, MODELS + 'face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      { numFaces: 1, outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true });
    if (!faceT) throw new Error('Не удалось запустить трекинг лица');
    cameraButton.textContent = 'Камера включена';
    await loadPose();
    if (armsOn) await loadHands();
  } catch (e) {
    console.error(e);
    stopCamera();
    alert(CAMERA_ERRORS[e?.name] ?? 'Не получилось включить камеру или трекинг. Подробности в консоли (F12).');
    cameraButton.disabled = false;
    cameraButton.textContent = 'Включить камеру';
  }
};

// ======================= ПОВОРОТЫ КОСТЕЙ =======================
const tmpQ = new THREE.Quaternion(), worldQ = new THREE.Quaternion(), localQ = new THREE.Quaternion();
const parentQ = new THREE.Quaternion(), identityQ = new THREE.Quaternion(), euler = new THREE.Euler();

// Плавно ведёт кость к цели, но не быстрее MAX_STEP за кадр
function steer(node, target, smooth) {
  if (!node || !Number.isFinite(target.x + target.y + target.z + target.w)) return;
  tmpQ.copy(node.quaternion).slerp(target, smooth);
  node.quaternion.rotateTowards(tmpQ, MAX_STEP);
}
// Поворот задан в мире, повороты родителей учитываются сами
function steerWorld(node, qWorld, smooth) {
  if (!node?.parent) return;
  node.parent.getWorldQuaternion(parentQ);
  localQ.copy(parentQ).invert().multiply(qWorld).multiply(vrm.scene.quaternion);
  steer(node, localQ, smooth);
}
const bone = (name) => vrm.humanoid.getNormalizedBoneNode(name);

// ======================= ЛИЦО =======================
function setExpr(name, target) {
  const e = vrm.expressionManager, cur = e.getValue(name) ?? 0;
  e.setValue(name, cur + (clamp01(target) - cur) * 0.5);
}
function setMorph(name, target) {
  const meshes = faceMeshes.filter((m) => m.morphTargetDictionary[name] !== undefined);
  if (!meshes.length) return;
  const cur = meshes[0].morphTargetInfluences[meshes[0].morphTargetDictionary[name]] ?? 0;
  const next = cur + (clamp01(target) - cur) * 0.5;
  for (const m of meshes) m.morphTargetInfluences[m.morphTargetDictionary[name]] = next;
}
function level(shapes, name, range) {
  const v = ((shapes[name] ?? 0) - (neutral[name] ?? 0)) / range;
  return Math.max(0, v - DEAD) / (1 - DEAD);
}

let jawPeak = 0.08;
function jawLevel(shapes) {
  const v = Math.max(0, (shapes.jawOpen ?? 0) - Math.min(neutral.jawOpen ?? 0, 0.03));
  jawPeak = Math.max(v, jawPeak * 0.99, 0.08);
  return clamp01((v - 0.02) / (jawPeak - 0.02));
}

const browDebug = { Left: [0, 0], Right: [0, 0] };
function applyBrows(shapes) {
  const L = (n) => level(shapes, n, RANGE.brow);
  const inner = L('browInnerUp');

  if (!browSplit) {
    const up = (L('browOuterUpLeft') + L('browOuterUpRight')) / 2;
    setMorph('Fcl_BRW_Surprised', up);
    setMorph('Fcl_BRW_Angry', (L('browDownLeft') + L('browDownRight')) / 2);
    setMorph('Fcl_BRW_Sorrow', (inner - up) * SORROW_GAIN);
    return;
  }

  const side = {};
  for (const m of ['Left', 'Right']) {
    const mp = mpSide(m, BROW_SWAP);
    side[m] = { up: clamp01(L('browOuterUp' + mp) * BROW_UP), down: clamp01(L('browDown' + mp) * BROW_DOWN) };
  }
  const sorrowBase = clamp01((inner - Math.max(side.Left.up, side.Right.up)) * SORROW_GAIN);

  for (const m of ['Left', 'Right']) {
    const me = side[m], other = side[m === 'Left' ? 'Right' : 'Left'];
    const down = clamp01(Math.max(me.down, Math.max(0, other.up - me.up - BROW_ASYM_DEAD) * BROW_ASYM));
    const net = me.up - down, up = Math.max(0, net), frown = Math.max(0, -net);
    const sfx = m === 'Left' ? '_L' : '_R';
    setMorph('Fcl_BRW_Surprised' + sfx, up);
    setMorph('Fcl_BRW_Angry' + sfx, frown);
    setMorph('Fcl_BRW_Sorrow' + sfx, sorrowBase * (1 - frown));
    browDebug[m] = [up, frown];
  }
}

// ======================= ВЗГЛЯД =======================
const gazeTarget = { yaw: 0, pitch: 0 }, gaze = { yaw: 0, pitch: 0 };
const peaks = { xp: GAZE_MIN_PEAK.x, xn: GAZE_MIN_PEAK.x, yp: GAZE_MIN_PEAK.y, yn: GAZE_MIN_PEAK.y };

function normGaze(v, axis) {
  const key = axis + (v >= 0 ? 'p' : 'n');
  peaks[key] = Math.min(GAZE_MAX_PEAK[axis], Math.max(Math.abs(v), peaks[key] * GAZE_PEAK_DECAY, GAZE_MIN_PEAK[axis]));
  return limit(v / peaks[key], 1);
}

function updateGaze(shapes, blinking, t) {
  if (blinking) return;   // при моргании взгляд замораживается (иначе глаза закатываются)
  const v = (n) => (shapes[n] ?? 0) - (neutral[n] ?? 0);
  const horizontal = (v('eyeLookOutLeft') + v('eyeLookInRight') - v('eyeLookInLeft') - v('eyeLookOutRight')) / 2;
  const vertical = (v('eyeLookUpLeft') + v('eyeLookUpRight') - v('eyeLookDownLeft') - v('eyeLookDownRight')) / 2;
  const hx = dead(filt('gx', horizontal, t, GAZE_FILTER_CUTOFF), GAZE_DEAD.x);
  const hy = dead(filt('gy', vertical, t, GAZE_FILTER_CUTOFF), GAZE_DEAD.y);
  gazeTarget.yaw = normGaze(hx, 'x') * GAZE_MAX * GAZE_YAW_SIGN * mir();
  gazeTarget.pitch = normGaze(hy, 'y') * GAZE_MAX * GAZE_PITCH_SIGN;
}

function applyGaze(faceLost) {
  if (!vrm.lookAt) return;
  if (faceLost) gazeTarget.yaw = gazeTarget.pitch = 0;
  gaze.yaw += (gazeTarget.yaw - gaze.yaw) * GAZE_SMOOTH;
  gaze.pitch += (gazeTarget.pitch - gaze.pitch) * GAZE_SMOOTH;
  vrm.lookAt.yaw = gaze.yaw;
  vrm.lookAt.pitch = gaze.pitch;
}

// ======================= ГОЛОВА =======================
const headTarget = { yaw: 0, pitch: 0, roll: 0 };
const mat = new THREE.Matrix4(), tmpPos = new THREE.Vector3(), tmpScale = new THREE.Vector3(), tmpQuat = new THREE.Quaternion();

function headAngles(result) {
  const m = result.facialTransformationMatrixes?.[0];
  if (!m) return null;
  mat.fromArray(m.data).decompose(tmpPos, tmpQuat, tmpScale);
  euler.setFromQuaternion(tmpQuat, 'YXZ');
  return { yaw: euler.y, pitch: euler.x, roll: euler.z };
}

function updateHeadTarget(shapes) {
  const yaw = shapes.headYaw - (neutral.headYaw ?? 0);
  const pitch = shapes.headPitch - (neutral.headPitch ?? 0);
  const roll = shapes.headRoll - (neutral.headRoll ?? 0);
  if (Math.abs(yaw) > MAX_VALID.yaw || Math.abs(pitch) > MAX_VALID.pitch || Math.abs(roll) > MAX_VALID.roll) return;
  headTarget.yaw = limit(yaw * YAW_SIGN * mir(), MAX_ANGLE.yaw);
  headTarget.pitch = limit(pitch * PITCH_SIGN, MAX_ANGLE.pitch);
  headTarget.roll = limit(roll * ROLL_SIGN * mir(), MAX_ANGLE.roll);
}

function applyHead(smooth) {
  for (const [name, k] of [['neck', NECK_SHARE], ['head', 1]]) {
    euler.set(headTarget.pitch * k, headTarget.yaw * k, headTarget.roll * k, 'YXZ');
    steerWorld(bone(name), worldQ.setFromEuler(euler), smooth);
  }
}

// ======================= ТЕЛО: ПЛЕЧИ И ТОРС =======================
function readBody() {
  const p = pose.image, w = pose.world;
  if (!vis(p[11]) || !vis(p[12])) return null;
  const ears = [7, 8].filter((i) => vis(p[i]));
  if (!ears.length) return null;
  const refY = ears.reduce((s, i) => s + p[i].y, 0) / ears.length;
  const W = video.videoWidth, H = video.videoHeight;
  const width = Math.hypot((p[11].x - p[12].x) * W, (p[11].y - p[12].y) * H);
  if (width < 20) return null;
  return {
    gapLeft: ((p[11].y - refY) * H) / width,
    gapRight: ((p[12].y - refY) * H) / width,
    depthLeft: w[11].z,
    depthRight: w[12].z,
    torsoRoll: Math.atan2((p[11].y - p[12].y) * H, (p[11].x - p[12].x) * W),
    torsoYaw: Math.atan2(w[11].z - w[12].z, w[11].x - w[12].x)
  };
}

function updatePose() {
  const t = performance.now();
  const r = poseT.detectForVideo(video, t);
  if (r.landmarks.length && r.worldLandmarks.length) {
    pose = { image: smoothPts(r.landmarks[0], 'i', t), world: smoothPts(r.worldLandmarks[0], 'w', t) };
    body = readBody();
    if (body) for (const k of BODY_KEYS) base[k] ??= body[k];
  } else {
    pose = body = null;
    resetFilters('i');
    resetFilters('w');
  }
}

function applyTorso() {
  let yaw = 0, roll = 0;
  if (body && base.torsoRoll !== undefined) {
    roll = limit(dead(body.torsoRoll - base.torsoRoll, TORSO_DEAD) * TORSO_ROLL_SIGN * mir(), TORSO_MAX.roll);
    yaw = limit(dead(body.torsoYaw - base.torsoYaw, TORSO_DEAD) * TORSO_YAW_SIGN * mir(), TORSO_MAX.yaw);
  }
  torsoBones.forEach((b, i) => {
    const share = (i + 1) / torsoBones.length;
    euler.set(0, yaw * share, roll * share, 'YXZ');
    steerWorld(b, worldQ.setFromEuler(euler), TORSO_SMOOTH);
  });
}

function applyShoulders() {
  for (const [side, name, own] of [[1, 'leftShoulder', 'Left'], [-1, 'rightShoulder', 'Right']]) {
    const b = bone(name);
    if (!b) continue;
    const src = mirror ? (own === 'Left' ? 'Right' : 'Left') : own;
    let up = 0, fwd = 0;
    if (body && base['gap' + src] !== undefined) {
      up = limit(dead(base['gap' + src] - body['gap' + src], SHOULDER_DEAD) * SHOULDER_GAIN, SHOULDER_MAX);
      fwd = limit(dead((body['depth' + src] - base['depth' + src]) * DEPTH_SIGN, DEPTH_DEAD) * DEPTH_GAIN, DEPTH_MAX);
    }
    const f = vrmFlip();
    euler.set(0, side * fwd * f, side * up * SHOULDER_SIGN * f, 'YXZ');
    steer(b, worldQ.setFromEuler(euler), SHOULDER_SMOOTH);
  }
}

// ======================= РУКИ =======================
const dirUpper = new THREE.Vector3(), dirLower = new THREE.Vector3(), restDir = new THREE.Vector3(), wantDir = new THREE.Vector3();
const upperQ = new THREE.Quaternion(), lowerQ = new THREE.Quaternion();

function setDirection(a, b, out) {
  return out.set((b.x - a.x) * ARM_SIGN.x * mir(), (b.y - a.y) * ARM_SIGN.y, (b.z - a.z) * ARM_SIGN.z);
}
function dirToQuat(side, dir, out) {
  if (dir.lengthSq() < 1e-6) return out.identity();
  return out.setFromUnitVectors(restDir.set(side, 0, 0), wantDir.copy(dir).normalize());
}

function applyArm(side) {
  const upper = bone(side === 1 ? 'leftUpperArm' : 'rightUpperArm');
  const lower = bone(side === 1 ? 'leftLowerArm' : 'rightLowerArm');
  if (!upper || !lower) return;

  const [s, e, w] = (mirror ? side === 1 : side === -1) ? [12, 14, 16] : [11, 13, 15];
  let upperOk = false, lowerOk = false;
  if (pose && armsOn) {
    const img = pose.image, wd = pose.world;
    upperOk = vis(img[s]) && vis(img[e]);
    lowerOk = upperOk && vis(img[w]);
    if (upperOk) setDirection(wd[s], wd[e], dirUpper);
    if (lowerOk) setDirection(wd[e], wd[w], dirLower);
  }
  if (!upperOk) dirUpper.set(side * ARM_REST_OUT, -1, 0);
  if (!lowerOk) dirLower.copy(dirUpper);

  steerWorld(upper, dirToQuat(side, dirUpper, upperQ), ARM_SMOOTH);
  steerWorld(lower, dirToQuat(side, dirLower, lowerQ), ARM_SMOOTH);
}

// ======================= КИСТИ И ПАЛЬЦЫ =======================
const FINGERS = [['Index', 5], ['Middle', 9], ['Ring', 13], ['Little', 17]];
const vA = new THREE.Vector3(), vB = new THREE.Vector3(), vF = new THREE.Vector3(), vS = new THREE.Vector3(), vU = new THREE.Vector3();
const basis = new THREE.Matrix4(), detectedQ = new THREE.Quaternion(), restQ = new THREE.Quaternion();
const forearmQ = new THREE.Quaternion(), sceneInvQ = new THREE.Quaternion(), relQ = new THREE.Quaternion(), limQ = new THREE.Quaternion();

function bendAt(a, b, c) {
  vA.set(b.x - a.x, b.y - a.y, b.z - a.z);
  vB.set(c.x - b.x, c.y - b.y, c.z - b.z);
  return vA.lengthSq() < 1e-10 || vB.lengthSq() < 1e-10 ? 0 : vA.angleTo(vB);
}
const curl = (angle, max) => Math.min(max, dead(Math.max(0, angle), FINGER_DEAD) * FINGER_GAIN);

function setFinger(name, z, y = 0) {
  euler.set(0, y, z, 'XYZ');
  const b = bone(name);
  if (b) steer(b, worldQ.setFromEuler(euler), FINGER_SMOOTH);
}

function applyFingers(side, w) {
  const pre = side === 1 ? 'left' : 'right';
  for (const [name, m] of FINGERS) {
    const p = w ? curl(bendAt(w[0], w[m], w[m + 1]), FINGER_MAX.proximal) : FINGER_REST;
    const i = w ? curl(bendAt(w[m], w[m + 1], w[m + 2]), FINGER_MAX.intermediate) : FINGER_REST;
    const d = w ? curl(bendAt(w[m + 1], w[m + 2], w[m + 3]), FINGER_MAX.distal) : FINGER_REST * 0.7;
    setFinger(pre + name + 'Proximal', -side * p * FINGER_SIGN);
    setFinger(pre + name + 'Intermediate', -side * i * FINGER_SIGN);
    setFinger(pre + name + 'Distal', -side * d * FINGER_SIGN);
  }
  const t0 = w ? curl(bendAt(w[0], w[1], w[2]), 0.7) * 0.6 : 0;
  const t1 = w ? curl(bendAt(w[1], w[2], w[3]), 1.0) : 0;
  const t2 = w ? curl(bendAt(w[2], w[3], w[4]), 1.2) : 0;
  [['ThumbMetacarpal', t0], ['ThumbProximal', t1], ['ThumbDistal', t2]].forEach(([n, t]) =>
    setFinger(pre + n, -side * t * 0.5 * FINGER_SIGN, side * t * THUMB_SIGN));
}

function handQuat(side, w, out) {
  setDirection(w[0], w[9], vF);
  setDirection(w[5], w[17], vS);
  if (vF.lengthSq() < 1e-8 || vS.lengthSq() < 1e-8) return false;
  vF.normalize();
  vS.addScaledVector(vF, -vS.dot(vF));
  if (vS.lengthSq() < 1e-8) return false;
  vS.normalize();
  if (HAND_FLIP[side === 1 ? 'left' : 'right']) vS.negate();
  vU.crossVectors(vS, vF);
  detectedQ.setFromRotationMatrix(basis.makeBasis(vF, vU, vS));

  vF.set(side, 0, 0);
  vS.set(0, 0, -1);
  vU.crossVectors(vS, vF);
  restQ.setFromRotationMatrix(basis.makeBasis(vF, vU, vS)).invert();
  out.copy(detectedQ).multiply(restQ);
  return true;
}

function limitToForearm(lower, q) {
  lower.getWorldQuaternion(forearmQ).multiply(sceneInvQ.copy(vrm.scene.quaternion).invert());
  relQ.copy(forearmQ).invert().multiply(q);
  if (relQ.w < 0) relQ.set(-relQ.x, -relQ.y, -relQ.z, -relQ.w);
  const angle = 2 * Math.acos(Math.min(1, relQ.w));
  if (angle > HAND_MAX_BEND) relQ.copy(limQ.identity().slerp(relQ, HAND_MAX_BEND / angle));
  q.copy(forearmQ).multiply(relQ);
}

function applyHand(side) {
  const pre = side === 1 ? 'left' : 'right';
  const data = armsOn ? hands[pre] : null;
  const hand = bone(pre + 'Hand'), lower = bone(pre + 'LowerArm');
  if (hand) {
    if (data && lower && handQuat(side, data.world, worldQ)) {
      limitToForearm(lower, worldQ);
      steerWorld(hand, worldQ, HAND_SMOOTH);
    } else {
      steer(hand, identityQ, HAND_SMOOTH);
    }
  }
  applyFingers(side, data?.world ?? null);
}

let handFrame = 0;
function updateHands() {
  if (++handFrame % HAND_EVERY) return;
  const t = performance.now();
  const r = handT.detectForVideo(video, t);
  const found = r.landmarks.map((image, i) => ({ image, world: r.worldLandmarks[i] }))
    .sort((a, b) => a.image[0].x - b.image[0].x);

  let left = null, right = null;
  if (found.length >= 2) {
    const first = found[0], last = found[found.length - 1];
    left = mirror ? first : last;
    right = mirror ? last : first;
  } else if (found.length === 1) {
    const h = found[0];
    let onImageLeft = h.image[0].x < 0.5;
    if (pose && vis(pose.image[15]) && vis(pose.image[16])) {
      const d = (p) => Math.hypot(h.image[0].x - p.x, h.image[0].y - p.y);
      onImageLeft = d(pose.image[16]) < d(pose.image[15]);
    }
    if (mirror ? onImageLeft : !onImageLeft) left = h; else right = h;
  }

  for (const [side, h, key] of [['left', left, 'hL'], ['right', right, 'hR']]) {
    if (h) hands[side] = { world: smoothPts(h.world, key + 'w', t) };
    else { hands[side] = null; resetFilters(key); }
  }
}

// ======================= ПРИМЕНЯЕМ ЛИЦО =======================
let debugCounter = 0, lastFaceTime = 0;

function applyFace(result) {
  const shapes = {};
  for (const c of result.faceBlendshapes[0].categories) shapes[c.categoryName] = c.score;
  const head = headAngles(result);
  if (head) Object.assign(shapes, { headYaw: head.yaw, headPitch: head.pitch, headRoll: head.roll });
  if (body) Object.assign(shapes, body);

  if (calibration) {
    for (const k in shapes) {
      calibration.sums[k] = (calibration.sums[k] ?? 0) + shapes[k];
      calibration.counts[k] = (calibration.counts[k] ?? 0) + 1;
    }
    if (performance.now() > calibration.end) {
      for (const k in calibration.sums) {
        const avg = calibration.sums[k] / calibration.counts[k];
        if (BODY_KEYS.includes(k)) base[k] = avg; else neutral[k] = avg;
      }
      localStorage.setItem('neutral', JSON.stringify(neutral));
      calibration = null;
      calibrateButton.textContent = 'Калибровка';
    }
  }

  if (++debugCounter % 10 === 0) {
    debugBox.textContent =
      ['eyeBlinkLeft', 'eyeBlinkRight', 'jawOpen', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'browDownLeft', 'browDownRight']
        .map((n) => n + ': ' + fmt(shapes[n]) + ' (покой ' + fmt(neutral[n]) + ')').join('\n') +
      '\nзеркало: ' + (mirror ? 'вкл' : 'выкл') +
      '\nтело: ' + (body ? 'видно' : 'не видно') +
      '\nкисти L/R: ' + (hands.left ? 'да' : 'нет') + ' / ' + (hands.right ? 'да' : 'нет') + (armsOn ? '' : ' (руки выкл)') +
      '\nраздельные брови: ' + (browSplit ? 'да' : 'нет') +
      '\nброви L вверх/вниз: ' + fmt(browDebug.Left[0]) + ' / ' + fmt(browDebug.Left[1]) +
      '\nброви R вверх/вниз: ' + fmt(browDebug.Right[0]) + ' / ' + fmt(browDebug.Right[1]) +
      '\nlookAt: ' + (vrm.lookAt ? 'да' : 'нет') + ', взгляд ' + gaze.yaw.toFixed(0) + ' / ' + gaze.pitch.toFixed(0) +
      '\nкостей торса: ' + torsoBones.length;
  }

  const adj = { ...shapes };
  for (const s of ['Left', 'Right']) {
    const down = Math.max(0, (shapes['eyeLookDown' + s] ?? 0) - (neutral['eyeLookDown' + s] ?? 0));
    adj['eyeBlink' + s] = (shapes['eyeBlink' + s] ?? 0) - GAZE_BLINK_COMP * down;
  }
  setExpr('blinkLeft', level(adj, 'eyeBlink' + mpSide('Left'), RANGE.blink));
  setExpr('blinkRight', level(adj, 'eyeBlink' + mpSide('Right'), RANGE.blink));
  setExpr('aa', jawLevel(shapes));
  setExpr('ou', level(shapes, 'mouthPucker', RANGE.pucker));
  setMorph('Fcl_MTH_Joy', (level(shapes, 'mouthSmileLeft', RANGE.smile) + level(shapes, 'mouthSmileRight', RANGE.smile)) / 2);
  applyBrows(shapes);

  const blinking = Math.max(level(shapes, 'eyeBlinkLeft', RANGE.blink), level(shapes, 'eyeBlinkRight', RANGE.blink)) > BLINK_HOLD;
  updateGaze(shapes, blinking, performance.now());
  if (head) updateHeadTarget(shapes);
}

// ======================= ГЛАВНЫЙ ЦИКЛ =======================
const clock = new THREE.Clock();
let lastVideoTime = -1, poseFrame = 0, errorShown = false;

function track() {
  if (!(faceT && video.readyState >= 2 && video.currentTime !== lastVideoTime)) return;
  lastVideoTime = video.currentTime;
  if (poseT && ++poseFrame % POSE_EVERY === 0) updatePose();
  if (armsOn && handT) updateHands();
  const result = faceT.detectForVideo(video, performance.now());
  if (vrm && result.faceBlendshapes.length) {
    lastFaceTime = performance.now();
    applyFace(result);
  }
}

function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();

  try { track(); } catch (e) {
    if (!errorShown) { console.error('Ошибка трекинга:', e); errorShown = true; }
  }

  if (vrm) {
    const faceLost = faceT && performance.now() - lastFaceTime > LOST_DELAY;
    if (faceLost) headTarget.yaw = headTarget.pitch = headTarget.roll = 0;

    applyTorso();
    applyShoulders();
    applyHead(faceLost ? 0.05 : HEAD_SMOOTH);
    applyArm(1); applyArm(-1);
    applyHand(1); applyHand(-1);
    applyGaze(faceLost);
    vrm.update(delta);
  }
  renderer.render(scene, camera);
}
animate();
