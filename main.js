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
// Если у кисти видна ладонь вместо тыльной
