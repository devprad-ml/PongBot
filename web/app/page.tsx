'use client';

import { useEffect, useRef, useState } from 'react';

const WIDTH = 800;
const HEIGHT = 600;
const PAD_W = 10;
const PAD_H = 90;
const BALL_SIZE = 12;
const PAD_SPEED = 6;
const MAX_SPEED = 12;
const SCORE_LIMIT = 10;

type GameState = 'menu' | 'playing' | 'gameover';

export default function PongGame() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sessionRef = useRef<any>(null);
  const gameRef = useRef({
    leftY: HEIGHT / 2 - PAD_H / 2,
    rightY: HEIGHT / 2 - PAD_H / 2,
    ballX: WIDTH / 2,
    ballY: HEIGHT / 2,
    ballVX: 4,
    ballVY: 4,
    leftScore: 0,
    rightScore: 0,
    keys: {} as Record<string, boolean>,
    animFrame: 0,
  });
  const [gameState, setGameState] = useState<GameState>('menu');
  const [winner, setWinner] = useState('');
  const [modelLoaded, setModelLoaded] = useState(false);
  const [loadError, setLoadError] = useState('');

  // Load ONNX model
  useEffect(() => {
    async function loadModel() {
      try {
        const ort = await import('onnxruntime-web');
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        const session = await ort.InferenceSession.create('/right_agent_single.onnx');
        sessionRef.current = session;
        setModelLoaded(true);
      } catch (e: any) {
        setLoadError('Failed to load AI model: ' + e.message);
      }
    }
    loadModel();
  }, []);

  // Key listeners
  useEffect(() => {
    const down = (e: KeyboardEvent) => { gameRef.current.keys[e.key] = true; };
    const up = (e: KeyboardEvent) => { gameRef.current.keys[e.key] = false; };
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    return () => { window.removeEventListener('keydown', down); window.removeEventListener('keyup', up); };
  }, []);

  function resetBall() {
    const g = gameRef.current;
    g.ballX = WIDTH / 2;
    g.ballY = HEIGHT / 2;
    g.ballVX = Math.random() > 0.5 ? 4 : -4;
    g.ballVY = Math.random() > 0.5 ? 4 : -4;
  }

  function startGame() {
    const g = gameRef.current;
    g.leftY = HEIGHT / 2 - PAD_H / 2;
    g.rightY = HEIGHT / 2 - PAD_H / 2;
    g.leftScore = 0;
    g.rightScore = 0;
    resetBall();
    setWinner('');
    setGameState('playing');
  }

  // Game loop
  useEffect(() => {
    if (gameState !== 'playing') return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    let running = true;

    async function getAIAction(): Promise<number> {
      if (!sessionRef.current) return 0;
      const g = gameRef.current;
      try {
        const ort = await import('onnxruntime-web');
        const state = new Float32Array([
          g.leftY / HEIGHT,
          g.rightY / HEIGHT,
          g.ballX / WIDTH,
          g.ballY / HEIGHT,
          g.ballVX / MAX_SPEED,
          g.ballVY / MAX_SPEED,
        ]);
        const tensor = new ort.Tensor('float32', state, [1, 6]);
        const result = await sessionRef.current.run({ input: tensor });
        const qValues = result.output.data as Float32Array;
        return qValues.indexOf(Math.max(...Array.from(qValues)));
      } catch {
        return 0;
      }
    }

    async function loop() {
      if (!running) return;
      const g = gameRef.current;

      // Human left paddle
      if (g.keys['w'] || g.keys['W']) g.leftY = Math.max(0, g.leftY - PAD_SPEED);
      if (g.keys['s'] || g.keys['S']) g.leftY = Math.min(HEIGHT - PAD_H, g.leftY + PAD_SPEED);

      // AI right paddle
      const action = await getAIAction();
      if (action === 1) g.rightY = Math.max(0, g.rightY - PAD_SPEED);
      if (action === 2) g.rightY = Math.min(HEIGHT - PAD_H, g.rightY + PAD_SPEED);

      // Ball movement
      g.ballX += g.ballVX;
      g.ballY += g.ballVY;

      // Bounce top/bottom
      if (g.ballY <= 0 || g.ballY >= HEIGHT - BALL_SIZE) g.ballVY *= -1;

      // Left paddle collision
      if (g.ballX <= 30 + PAD_W && g.ballY + BALL_SIZE >= g.leftY && g.ballY <= g.leftY + PAD_H && g.ballVX < 0) {
        g.ballVX *= -1.05;
        g.ballVX = Math.min(Math.abs(g.ballVX), MAX_SPEED) * Math.sign(g.ballVX);
      }
      // Right paddle collision
      if (g.ballX + BALL_SIZE >= WIDTH - 30 - PAD_W && g.ballY + BALL_SIZE >= g.rightY && g.ballY <= g.rightY + PAD_H && g.ballVX > 0) {
        g.ballVX *= -1.05;
        g.ballVX = Math.max(-MAX_SPEED, Math.min(MAX_SPEED, g.ballVX));
      }

      // Scoring
      if (g.ballX < 0) {
        g.rightScore++;
        resetBall();
      } else if (g.ballX > WIDTH) {
        g.leftScore++;
        resetBall();
      }

      // Check win
      if (g.leftScore >= SCORE_LIMIT || g.rightScore >= SCORE_LIMIT) {
        setWinner(g.leftScore >= SCORE_LIMIT ? 'You win! 🎉' : 'AI wins! 🤖');
        setGameState('gameover');
        return;
      }

      // Draw
      ctx.fillStyle = '#0a0a0a';
      ctx.fillRect(0, 0, WIDTH, HEIGHT);

      // Center line
      ctx.setLineDash([10, 10]);
      ctx.strokeStyle = '#222';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(WIDTH / 2, 0);
      ctx.lineTo(WIDTH / 2, HEIGHT);
      ctx.stroke();
      ctx.setLineDash([]);

      // Paddles
      ctx.shadowBlur = 15;
      ctx.shadowColor = '#00ff88';
      ctx.fillStyle = '#00ff88';
      ctx.fillRect(25, g.leftY, PAD_W, PAD_H);

      ctx.shadowColor = '#ff4466';
      ctx.fillStyle = '#ff4466';
      ctx.fillRect(WIDTH - 25 - PAD_W, g.rightY, PAD_W, PAD_H);

      // Ball
      ctx.shadowColor = '#ffffff';
      ctx.shadowBlur = 20;
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(g.ballX + BALL_SIZE / 2, g.ballY + BALL_SIZE / 2, BALL_SIZE / 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;

      // Scores
      ctx.font = 'bold 48px monospace';
      ctx.textAlign = 'center';
      ctx.fillStyle = '#00ff88';
      ctx.fillText(String(g.leftScore), WIDTH / 4, 60);
      ctx.fillStyle = '#ff4466';
      ctx.fillText(String(g.rightScore), (3 * WIDTH) / 4, 60);

      // Labels
      ctx.font = '12px monospace';
      ctx.fillStyle = '#444';
      ctx.fillText('YOU', WIDTH / 4, 80);
      ctx.fillText('AI', (3 * WIDTH) / 4, 80);

      if (running) g.animFrame = requestAnimationFrame(loop);
    }

    loop();
    return () => {
      running = false;
      cancelAnimationFrame(gameRef.current.animFrame);
    };
  }, [gameState]);

  return (
    <main style={{
      minHeight: '100vh',
      background: '#050505',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: 'monospace',
      color: '#fff',
      padding: '20px',
    }}>
      <h1 style={{ fontSize: 32, fontWeight: 800, letterSpacing: '0.2em', marginBottom: 8, color: '#00ff88' }}>
        PONGBOT
      </h1>
      <p style={{ color: '#444', fontSize: 13, marginBottom: 24, letterSpacing: '0.1em' }}>
        DQN-trained AI · Deep Q-Learning · PyTorch → ONNX
      </p>

      {loadError && (
        <div style={{ color: '#ff4466', marginBottom: 16, fontSize: 13 }}>{loadError}</div>
      )}

      <div style={{ position: 'relative' }}>
        <canvas
          ref={canvasRef}
          width={WIDTH}
          height={HEIGHT}
          style={{
            border: '1px solid #1a1a1a',
            borderRadius: 8,
            display: 'block',
            maxWidth: '100%',
          }}
        />

        {/* Menu overlay */}
        {gameState === 'menu' && (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center',
            background: 'rgba(5,5,5,0.92)', borderRadius: 8,
          }}>
            <div style={{ fontSize: 64, marginBottom: 16 }}>🏓</div>
            <h2 style={{ fontSize: 28, fontWeight: 700, marginBottom: 8, letterSpacing: '0.1em' }}>
              Human vs AI
            </h2>
            <p style={{ color: '#555', marginBottom: 32, fontSize: 13 }}>
              {modelLoaded ? 'AI model loaded ✓' : 'Loading AI model...'}
            </p>
            <div style={{ color: '#333', fontSize: 13, marginBottom: 32, textAlign: 'center', lineHeight: 2 }}>
              <span style={{ color: '#00ff88' }}>W / S</span> — move your paddle
            </div>
            <button
              onClick={startGame}
              disabled={!modelLoaded}
              style={{
                padding: '14px 48px', background: modelLoaded ? '#00ff88' : '#1a1a1a',
                color: modelLoaded ? '#000' : '#333', border: 'none', borderRadius: 6,
                fontSize: 16, fontWeight: 700, cursor: modelLoaded ? 'pointer' : 'not-allowed',
                letterSpacing: '0.1em', fontFamily: 'monospace',
              }}
            >
              {modelLoaded ? 'PLAY' : 'LOADING...'}
            </button>
          </div>
        )}

        {/* Game over overlay */}
        {gameState === 'gameover' && (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center',
            background: 'rgba(5,5,5,0.92)', borderRadius: 8,
          }}>
            <div style={{ fontSize: 64, marginBottom: 16 }}>
              {winner.includes('You') ? '🏆' : '🤖'}
            </div>
            <h2 style={{ fontSize: 32, fontWeight: 700, marginBottom: 32, letterSpacing: '0.05em' }}>
              {winner}
            </h2>
            <button
              onClick={startGame}
              style={{
                padding: '14px 48px', background: '#00ff88', color: '#000',
                border: 'none', borderRadius: 6, fontSize: 16, fontWeight: 700,
                cursor: 'pointer', letterSpacing: '0.1em', fontFamily: 'monospace',
              }}
            >
              PLAY AGAIN
            </button>
          </div>
        )}
      </div>

      <p style={{ color: '#222', fontSize: 11, marginTop: 20, letterSpacing: '0.05em' }}>
        First to {SCORE_LIMIT} points wins · AI runs entirely in your browser
      </p>
    </main>
  );
}