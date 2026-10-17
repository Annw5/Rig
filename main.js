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
const bodyButton = document.getElementById('bodyButton');
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

// ---------- Загрузчик VRM ----------
const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));

let currentVrm = null;
let faceMeshes = [];

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
      console.log('Кусков лица с формами:', faceMeshes.length);
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

// ---------- Настройки ----------
let neutral = {};
let calibration = null;

try {
  neutral = JSON.parse(localStorage.getItem('neutral')) ?? {};
} catch (error) {
  neutral = {};
}

// NEW: нулевые точки тела. Они не сохраняются между запусками
let base = {};
const BODY_KEYS = ['headX', 'headY', 'headZ', 'gapLeft', 'gapRight'];

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

// Голова
const YAW_SIGN = -1;
const PITCH_SIGN = 1;
const ROLL_SIGN = -1;
const HEAD_SHARE = 0.7;
const NECK_SHARE = 0.3;
const HEAD_SMOOTH = 0.4;
const MAX_ANGLE = { yaw: 1.0, pitch: 0.8, roll: 0.7 };

// Потеря лица
const MAX_VALID = { yaw: 1.4, pitch: 1.1, roll: 1.0 };
const LOST_DELAY = 400;
const RELAX_SMOOTH = 0.05;
let lastFaceTime = 0;

// NEW: положение модели в пространстве (из положения лица)
const POS_SIGN = { x: -1, y: 1, z: 1 };
const POS_GAIN = { x: 0.012, y: 0.012, z: 0.03 };
const POS_MAX = { x: 0.6, y: 0.4, z: 1.2 };
const POS_SMOOTH = 0.15;

// Руки
const ARM_SIGN = { x: -1, y: -1, z: -1 };
const POSE_VISIBLE = 0.5;
const ARM_SMOOTH = 0.3;
const ARM_REST_OUT = 0.25;
const POSE_EVERY = 2;

// NEW: плечи
const SHOULDER_SIGN = 1;
const SHOULDER_GAIN = 2.5;
const SHOULDER_DEAD = 0.02;
const SHOULDER_MAX = 0.5;
const SHOULDER_SMOOTH = 0.3;

// NEW: брови
const SORROW_GAIN = 1.0;

let debugCounter = 0;

// ---------- Камера и трекинг ----------
let faceLandmarker = null;
let poseLandmarker = null;
let visionFileset = null; // NEW
let poseLoading = false; // NEW
let latestPose = null;
let latestGaps = null; // NEW
let poseFrame = 0;

// NEW: включение и выключение трекинга рук и плеч
let bodyOn = localStorage.getItem('bodyOn') === '1';

function updateBodyButton() {
  bodyButton.textContent = 'Руки и плечи: ' + (bodyOn ? 'вкл' : 'выкл');
}
updateBodyButton();

async function loadPose() {
  if (!visionFileset || poseLandmarker || poseLoading) return;

  poseLoading = true;
  bodyButton.textContent = 'Загрузка тела...';
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
  updateBodyButton();
}

function unloadPose() {
  if (poseLandmarker) {
    poseLandmarker.close();
    poseLandmarker = null;
  }
  latestPose = null;
  latestGaps = null;
}

bodyButton.addEventListener('click', async () => {
  bodyOn = !bodyOn;
  localStorage.setItem('bodyOn', bodyOn ? '1' : '0');
  updateBodyButton();

  if (bodyOn) {
    await loadPose();
  } else {
    unloadPose();
  }
});

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

    if (bodyOn) {
      await loadPose();
    }
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
    calibration = { endTime: performance.now() + 2000, sums: {}, count: 0 };
    calibrateButton.textContent = 'Идёт замер...';
  }, 2000);
});

resetButton.addEventListener('click', () => {
  neutral = {};
  base = {};
  localStorage.removeItem('neutral');
});

// ---------- Помощники ----------
function clamp01(value) {
  return Math.min(1, Math.max(0, value));
}

function limit(value, max) {
  return Math.max(-max, Math.min(max, value));
}

function setSmooth(name, target) {
  const expressions = currentVrm.expressionManager;
  const current = expressions.getValue(name) ?? 0;
  const next = current + (clamp01(target) - current) * 0.5;
  expressions.setValue(name, next);
}

function setMorphSmooth(name, target) {
  if (faceMeshes.length === 0) return;

  const first = faceMeshes[0];
  const firstIndex = first.morphTargetDictionary[name];
  const current = first.morphTargetInfluences[firstIndex] ?? 0;
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
  const value = (shapes[name] - base0) / range;
  return Math.max(0, value - dead) / (1 - dead);
}

function jawLevel(shapes) {
  const jawBase = Math.min(neutral.jawOpen ?? 0, JAW_MAX_BASE);
  const value = Math.max(0, (shapes.jawOpen ?? 0) - jawBase);

  jawPeak = Math.max(value, jawPeak * JAW_DECAY, JAW_MIN_PEAK);

  return clamp01((value - DEAD_JAW) / (jawPeak - DEAD_JAW));
}

// ---------- Голова ----------
const tmpMatrix = new THREE.Matrix4();
const tmpPosition = new THREE.Vector3();
const tmpQuaternion = new THREE.Quaternion();
const tmpScale = new THREE.Vector3();
const tmpEuler = new THREE.Euler();
const targetQuaternion = new THREE.Quaternion();
const identityQuaternion = new THREE.Quaternion();

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
    x: tmpPosition.x, // NEW
    y: tmpPosition.y, // NEW
    z: tmpPosition.z // NEW
  };
}

function rotateBone(bone, x, y, z, share) {
  if (!bone) return;
  tmpEuler.set(x * share, y * share, z * share, 'YXZ');
  targetQuaternion.setFromEuler(tmpEuler);
  bone.quaternion.slerp(targetQuaternion, HEAD_SMOOTH);
}

function applyHead(shapes) {
  const humanoid = currentVrm.humanoid;
  const head = humanoid.getNormalizedBoneNode('head');
  const neck = humanoid.getNormalizedBoneNode('neck');

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

  const yaw = limit(rawYaw * YAW_SIGN, MAX_ANGLE.yaw);
  const pitch = limit(rawPitch * PITCH_SIGN, MAX_ANGLE.pitch);
  const roll = limit(rawRoll * ROLL_SIGN, MAX_ANGLE.roll);

  const flip = currentVrm.meta.metaVersion === '0' ? -1 : 1;

  rotateBone(head, pitch * flip, yaw, roll * flip, HEAD_SHARE);
  rotateBone(neck, pitch * flip, yaw, roll * flip, NECK_SHARE);
}

function relaxHead() {
  const humanoid = currentVrm.humanoid;
  for (const name of ['head', 'neck']) {
    const bone = humanoid.getNormalizedBoneNode(name);
    if (bone) {
      bone.quaternion.slerp(identityQuaternion, RELAX_SMOOTH);
    }
  }
}

// ---------- NEW: Положение модели (вперёд, назад, в стороны) ----------
const posTarget = new THREE.Vector3();

function applyBodyPosition(shapes) {
  base.headX ??= shapes.headX;
  base.headY ??= shapes.headY;
  base.headZ ??= shapes.headZ;

  posTarget.set(
    limit((shapes.headX - base.headX) * POS_SIGN.x * POS_GAIN.x, POS_MAX.x),
    limit((shapes.headY - base.headY) * POS_SIGN.y * POS_GAIN.y, POS_MAX.y),
    limit((shapes.headZ - base.headZ) * POS_SIGN.z * POS_GAIN.z, POS_MAX.z)
  );
  currentVrm.scene.position.lerp(posTarget, POS_SMOOTH);
}

// ---------- NEW: Плечи ----------
function getShoulderGaps() {
  if (!latestPose) return null;

  const p = latestPose.image;
  const visible = [7, 8, 11, 12].every((i) => (p[i].visibility ?? 0) > POSE_VISIBLE);
  if (!visible) return null;

  const w = video.videoWidth;
  const h = video.videoHeight;
  const width = Math.hypot((p[11].x - p[12].x) * w, (p[11].y - p[12].y) * h);
  if (width < 20) return null;

  return {
    left: ((p[11].y - p[7].y) * h) / width,
    right: ((p[12].y - p[8].y) * h) / width
  };
}

function applyShoulder(side, gap, baseGap) {
  const bone = currentVrm.humanoid.getNormalizedBoneNode(
    side === 1 ? 'leftShoulder' : 'rightShoulder'
  );
  if (!bone) return;

  const flip = currentVrm.meta.metaVersion === '0' ? -1 : 1;

  let angle = 0;
  if (gap !== undefined && baseGap !== undefined) {
    const diff = baseGap - gap;
    if (Math.abs(diff) > SHOULDER_DEAD) {
      angle = limit((diff - Math.sign(diff) * SHOULDER_DEAD) * SHOULDER_GAIN, SHOULDER_MAX);
    }
  }

  tmpEuler.set(0, 0, side * angle * SHOULDER_SIGN * flip);
  targetQuaternion.setFromEuler(tmpEuler);
  bone.quaternion.slerp(targetQuaternion, SHOULDER_SMOOTH);
}

function applyShoulders() {
  applyShoulder(1, latestGaps?.right, base.gapRight);
  applyShoulder(-1, latestGaps?.left, base.gapLeft);
}

// ---------- Руки ----------
const armRest = new THREE.Vector3();
const armWant = new THREE.Vector3();
const upperQ = new THREE.Quaternion();
const lowerQ = new THREE.Quaternion();
const upperInverse = new THREE.Quaternion();
const shoulderInverse = new THREE.Quaternion(); // NEW
const dirUpper = new THREE.Vector3();
const dirLower = new THREE.Vector3();

function setDirection(from, to, target) {
  target.set(
    (to.x - from.x) * ARM_SIGN.x,
    (to.y - from.y) * ARM_SIGN.y,
    (to.z - from.z) * ARM_SIGN.z
  );
}

function quaternionFromDirection(side, dir, flip, out) {
  armRest.set(side * flip, 0, 0);
  armWant.set(dir.x * flip, dir.y, dir.z * flip);
  if (armWant.lengthSq() < 0.000001) {
    out.identity();
    return;
  }
  armWant.normalize();
  out.setFromUnitVectors(armRest, armWant);
}

function applyArm(side, shoulderId, elbowId, wristId) {
  const humanoid = currentVrm.humanoid;
  const upperBone = humanoid.getNormalizedBoneNode(side === 1 ? 'leftUpperArm' : 'rightUpperArm');
  const lowerBone = humanoid.getNormalizedBoneNode(side === 1 ? 'leftLowerArm' : 'rightLowerArm');
  const shoulderBone = humanoid.getNormalizedBoneNode(side === 1 ? 'leftShoulder' : 'rightShoulder');
  if (!upperBone || !lowerBone) return;

  const flip = currentVrm.meta.metaVersion === '0' ? -1 : 1;

  let upperVisible = false;
  let lowerVisible = false;

  if (latestPose) {
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

  quaternionFromDirection(side, dirUpper, flip, upperQ);
  quaternionFromDirection(side, dirLower, flip, lowerQ);

  upperInverse.copy(upperQ).invert();
  lowerQ.premultiply(upperInverse);

  // NEW: учитываем, что плечо уже повёрнуто
  if (shoulderBone) {
    shoulderInverse.copy(shoulderBone.quaternion).invert();
    upperQ.premultiply(shoulderInverse);
  }

  upperBone.quaternion.slerp(upperQ, ARM_SMOOTH);
  lowerBone.quaternion.slerp(lowerQ, ARM_SMOOTH);
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
    latestGaps = getShoulderGaps();
    if (latestGaps) {
      base.gapLeft ??= latestGaps.left;
      base.gapRight ??= latestGaps.right;
    }
  } else {
    latestPose = null;
    latestGaps = null;
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
  if (latestGaps) {
    shapes.gapLeft = latestGaps.left;
    shapes.gapRight = latestGaps.right;
  }

  if (calibration) {
    for (const name in shapes) {
      calibration.sums[name] = (calibration.sums[name] ?? 0) + shapes[name];
    }
    calibration.count++;

    if (performance.now() > calibration.endTime) {
      for (const name in calibration.sums) {
        neutral[name] = calibration.sums[name] / calibration.count;
      }
      // NEW: данные тела уходят в base и не сохраняются
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
      'browInnerUp', 'browOuterUpLeft', 'browDownLeft',
      'headYaw', 'headPitch', 'headRoll'
    ];
    debugBox.textContent =
      names
        .map((n) => n + ': сейчас ' + (shapes[n] ?? 0).toFixed(2) + ', покой ' + (neutral[n] ?? 0).toFixed(2))
        .join('\n') +
      '\nпик рта: ' + jawPeak.toFixed(2) +
      '\nтело: ' + (latestPose ? 'видно' : 'не видно') +
      '\nдистанция: ' + (shapes.headZ ?? 0).toFixed(1) + ' см';
  }

  setSmooth('blinkLeft', level(shapes, 'eyeBlinkLeft', RANGE.blink));
  setSmooth('blinkRight', level(shapes, 'eyeBlinkRight', RANGE.blink));
  setSmooth('aa', jawLevel(shapes));
  setSmooth('ou', level(shapes, 'mouthPucker', RANGE.pucker));

  const smile =
    (level(shapes, 'mouthSmileLeft', RANGE.smile) +
      level(shapes, 'mouthSmileRight', RANGE.smile)) / 2;
  setMorphSmooth('Fcl_MTH_Joy', smile);

  // NEW: брови
  const browOuter =
    (level(shapes, 'browOuterUpLeft', RANGE.brow) +
      level(shapes, 'browOuterUpRight', RANGE.brow)) / 2;
  const browDown =
    (level(shapes, 'browDownLeft', RANGE.brow) +
      level(shapes, 'browDownRight', RANGE.brow)) / 2;
  const browInner = level(shapes, 'browInnerUp', RANGE.brow);
  const browSorrow = clamp01((browInner - browOuter) * SORROW_GAIN);

  setMorphSmooth('Fcl_BRW_Surprised', browOuter);
  setMorphSmooth('Fcl_BRW_Angry', browDown);
  setMorphSmooth('Fcl_BRW_Sorrow', browSorrow);

  if (headAngles) {
    applyHead(shapes);
    applyBodyPosition(shapes);
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

    const result = faceLandmarker.detectForVideo(video, performance.now());
    if (currentVrm && result.faceBlendshapes.length > 0) {
      lastFaceTime = performance.now();
      applyFace(result);
    }

    if (bodyOn && poseLandmarker) {
      updatePose();
    }
  }

  if (currentVrm && faceLandmarker && performance.now() - lastFaceTime > LOST_DELAY) {
    relaxHead();
  }

  if (currentVrm) {
    applyShoulders(); // NEW
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
