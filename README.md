# BodyLab

Track body composition by walking past a camera. BodyLab watches a network camera, measures your silhouette, estimates circumferences and body fat, and fits a realistic 3D body to you, clickable, with goals, logged inputs (workouts, food, sleep, activity), and an **inputs → outputs** view of how they move your body.

Everything runs locally. Video stays in memory; only measurements are stored.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and a camera that serves MJPEG or RTSP.

```sh
uv run app.py              # http://localhost:8765 (downloads the 30 MB pose model on first run)
uv run python test_bodylab.py && node test_body.mjs
```

Open the app, enter your height and camera URL under **Profile**, then stand in front of the camera:

1. Full body in frame, fitted clothes, arms ~30° out. Hold ~3 s → front scan saved.
2. Turn 90° for ~2 s → side scan adds real body depth (much better circumferences).

Env: `PORT` (8765), `COOLDOWN_MIN` (10, minimum minutes between saved scans), `BODYLAB_DB` (default `data/bodylab.db`).

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

## iPhone: Apple Health sync

`ios/` is a small SwiftUI app that sends the last 60 days of weight, calories, protein, sleep, active energy and workouts to BodyLab: on open, and in the background whenever Health gets new data. Re-syncs update records instead of duplicating them.

```sh
tailscale serve --bg --https=8443 http://127.0.0.1:8765   # tailnet-only HTTPS; only /api/sync (token) answers remotely
cd ios && xcodegen && open BodyLabSync.xcodeproj
```

Pick your Team under Signing & Capabilities, run it on your iPhone, enter the server and token shown in BodyLab → Profile, then tap **Connect Apple Health**.

## How it works

- [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) finds 33 landmarks and a person mask per frame.
- Widths are measured across the mask at anatomical levels (neck, chest, navel, hips, limbs), scaled by your height. A side view supplies depths; otherwise typical depth ratios are assumed.
- Circumferences use an ellipse model; body fat uses the US Navy formula (height, neck, waist, + hips for women).
- Optional: tape-measure any part once to calibrate it; past and future scans are corrected.
- The 3D body is MakeHuman's CC0 base mesh (`tools/build_body.py` → `static/body.bin`), fitted in the browser: torso width/depth per level, limbs radially around their bones, heights to your landmarks, blended by the rig's skin weights.

Expect roughly ±3–4 body-fat points without calibration. Clothing, posture and lighting matter more than the math.

## License

MIT. Bundles [three.js](https://threejs.org) (MIT) and a body mesh derived from [MakeHuman](http://www.makehumancommunity.org) assets (CC0); downloads the MediaPipe pose model (Apache-2.0).
