import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

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

const renderer = new THREE.WebGLRenderer({ antialias: true });
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

// ---------- Загрузчик VRM ----------
const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));

let currentVrm = null;

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
      console.log('Выражения модели:', names);
      exprBox.textContent = 'Выражения модели (' + names.length + '):\n' + names.join(', ');
      const morphNames = [];
      vrm.scene.traverse((object) => {
        if (object.isMesh && object.morphTargetDictionary) {
          morphNames.push(object.name + ': ' + Object.keys(object.morphTargetDictionary).join(', '));
        }
      });
      console.log('Формы сетки:', morphNames);
      exprBox.textContent += '\n\nФормы сетки:\n' + morphNames.join('\n');
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
const REST = {
  eyeBlinkLeft: 0.14,
  eyeBlinkRight: 0.10
};

let neutral = { ...REST };
let calibration = null;

try {
  const saved = JSON.parse(localStorage.getItem('neutral')) ?? {};
  neutral = { ...saved, ...REST };
} catch (error) {
  neutral = { ...REST };
}

const RANGE = {
  blink: 0.5,
  jaw: 0.4,
  pucker: 0.5,
  smile: 0.5
};
const HAPPY_MAX = 0.5;
const DEAD = 0.08;

let debugCounter = 0;

// ---------- Камера и трекинг лица ----------
let faceLandmarker = null;

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

    faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU'
      },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true
    });

    cameraButton.textContent = 'Камера включена';
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
  neutral = { ...REST };
  localStorage.removeItem('neutral');
});

// ---------- Помощники ----------
function clamp01(value) {
  return Math.min(1, Math.max(0, value));
}

function hasExpression(name) {
  return currentVrm.expressionManager.getExpression(name) != null;
}

function setSmooth(name, target) {
  const expressions = currentVrm.expressionManager;
  const current = expressions.getValue(name) ?? 0;
  const next = current + (clamp01(target) - current) * 0.5;
  expressions.setValue(name, next);
}

function level(shapes, name, range) {
  const base = neutral[name] ?? 0;
  const value = (shapes[name] - base) / range;
  return Math.max(0, value - DEAD) / (1 - DEAD);
}

// ---------- Применяем лицо к модели ----------
function applyFace(result) {
  const shapes = {};
  for (const item of result.faceBlendshapes[0].categories) {
    shapes[item.categoryName] = item.score;
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
      Object.assign(neutral, REST);
      localStorage.setItem('neutral', JSON.stringify(neutral));
      calibration = null;
      calibrateButton.textContent = 'Калибровка';
    }
  }

  debugCounter++;
  if (debugCounter % 10 === 0) {
    const names = [
      'eyeBlinkLeft', 'eyeBlinkRight', 'jawOpen',
      'mouthPucker', 'mouthSmileLeft', 'mouthSmileRight'
    ];
    debugBox.textContent = names
      .map((n) => n + ': сейчас ' + shapes[n].toFixed(2) + ', покой ' + (neutral[n] ?? 0).toFixed(2))
      .join('\n');
  }

  setSmooth('blinkLeft', level(shapes, 'eyeBlinkLeft', RANGE.blink));
  setSmooth('blinkRight', level(shapes, 'eyeBlinkRight', RANGE.blink));
  setSmooth('aa', level(shapes, 'jawOpen', RANGE.jaw));
  setSmooth('ou', level(shapes, 'mouthPucker', RANGE.pucker));

  const smileLeft = level(shapes, 'mouthSmileLeft', RANGE.smile);
  const smileRight = level(shapes, 'mouthSmileRight', RANGE.smile);

  if (hasExpression('mouthSmileLeft') && hasExpression('mouthSmileRight')) {
    setSmooth('mouthSmileLeft', smileLeft);
    setSmooth('mouthSmileRight', smileRight);
  } else {
    setSmooth('happy', ((smileLeft + smileRight) / 2) * HAPPY_MAX);
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
      applyFace(result);
    }
  }

  if (currentVrm) {
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
