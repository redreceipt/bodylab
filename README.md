# BodyLab

Track body composition by walking past a camera. BodyLab watches a network camera, measures your silhouette, estimates circumferences and body fat, and shows a clickable 3D body with goals, logged inputs (workouts, food, sleep, activity), and an **inputs → outputs** view of how they move your body.

Everything runs locally. Video stays in memory; only measurements are stored.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and a camera that serves MJPEG or RTSP.

```sh
uv run app.py              # http://localhost:8765 (downloads the 30 MB pose model on first run)
uv run python test_measure.py
```

Open the app, enter your height and camera URL under **Profile**, then stand in front of the camera:

1. Full body in frame, fitted clothes, arms ~30° out. Hold ~3 s → front scan saved.
2. Turn 90° for ~2 s → side scan adds real body depth (much better circumferences).

Env: `PORT` (8765), `COOLDOWN_MIN` (10, minimum minutes between saved scans).

## Raspberry Pi camera

`pi/camstream.py` serves `http://<pi-tailscale-ip>:8000/stream.mjpg` using picamera2, bound to the Pi's Tailscale address only.

```sh
PI=you@raspberrypi
tailscale ssh $PI 'mkdir -p ~/bodylab-cam ~/.config/systemd/user'
tailscale ssh $PI 'cat > ~/bodylab-cam/camstream.py' < pi/camstream.py
tailscale ssh $PI 'cat > ~/.config/systemd/user/bodylab-cam.service' < pi/bodylab-cam.service
tailscale ssh $PI 'loginctl enable-linger && systemctl --user daemon-reload && systemctl --user enable --now bodylab-cam'
```

Mounted sideways? Set **Camera rotation** in Profile.

## How it works

- [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) finds 33 landmarks and a person mask per frame.
- Widths are measured across the mask at anatomical levels (neck, chest, navel, hips, limbs), scaled by your height. A side view supplies depths; otherwise typical depth ratios are assumed.
- Circumferences use an ellipse model; body fat uses the US Navy formula (height, neck, waist, + hips for women).
- Optional: tape-measure any part once to calibrate it; past and future scans are corrected.

Expect roughly ±3–4 body-fat points without calibration. Clothing, posture and lighting matter more than the math.

## License

MIT. Bundles [three.js](https://threejs.org) (MIT); downloads the MediaPipe pose model (Apache-2.0).
