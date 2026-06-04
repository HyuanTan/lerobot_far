```bash
    python examples/rtc/eval_with_real_robot.py \
        --policy.path=jadenovalight/smolvla_pick-place_v2.4 \
        --policy.device=cuda \
        --fps=20 \
        --rtc.enabled=true \
        --rtc.execution_horizon=10 \
        --action_queue_size_to_get_new_actions=50 \
        --robot.type=so100_follower \
        --robot.port=/dev/ttyACM_so101follower \
        --robot.id=cse_so101follower \
        --robot.cameras="{top: {type: opencv, index_or_path: '/dev/videotop', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, wrist: {type: opencv, index_or_path: '/dev/videowrist', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, front: {type: opencv, index_or_path: '/dev/videofront', width: 640, height: 480, fps: 30, backend: 200, fourcc: MJPG}}" \
        --task="Pick up the yellow cube and put it into the box." \
        --duration=120



python examples/rtc/eval_real_robot_rtc_analysis.py \
        --policy.path=jadenovalight/smolvla_pick-place_v2.4 \
        --policy.device=cuda \
        --fps=20 \
        --rtc.enabled=true \
        --rtc.execution_horizon=10 \
        --action_queue_size_to_get_new_actions=50 \
        --robot.type=so100_follower \
        --robot.port=/dev/ttyACM_so101follower \
        --robot.id=cse_so101follower \
        --robot.cameras="{top: {type: opencv, index_or_path: '/dev/videotop', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, wrist: {type: opencv, index_or_path: '/dev/videowrist', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, front: {type: opencv, index_or_path: '/dev/videofront', width: 640, height: 480, fps: 30, backend: 200, fourcc: MJPG}}" \
        --task="Pick up the yellow cube and put it into the box." \
        --mock_robot=true \
        --duration=120


--mock_robot=true
```

Indexes diff is not equal to real delay. indexes_diff=20, real_delay=30
WARNING 2026-05-28 18:18:16 al_robot.py:425 [GET_ACTIONS] cfg.action_queue_size_to_get_new_actions Too small, It should be higher than inference delay + execution horizon.
