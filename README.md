# isaac-rl

## SceneFactory vehicle caveat

- The current SceneFactory Isaac Lab training path assumes one vehicle family is used across all worlds in a run.
- Different worlds may have different roads, spawns, and scene assignments, but the vehicle asset is assumed to share the same USD structure, joint layout, body layout, and proxy geometry everywhere.
- Variants that only change runtime properties such as tire friction or damping should preferably be handled as parameter overrides, not as different USDs per world.
- Supporting multiple per-world USD families would require a follow-up refactor of the current batched action/setup path.

## currrent working vehicle optimization

`nohup /home/yz8733/miniforge3/envs/isaac/bin/python -m src.student_vehicle_sysid   --headless   --teacher-dataset-manifest artifacts/physx_teacher_datasets/fwd_v1/manifest.json   --student-usd artifacts/student_vehicle_assets/vehicle_student/student_fwd_vehicle.usd   --output-dir artifacts/student_vehicle_sysid/fwd_v1_staged_cem_overnight_02   --search-mode staged   --optimizer cem   --random-search-trials 640   --random-search-seed 123   --cem-population-size 32   --cem-elite-fraction 0.20   --cem-initial-std-fraction 0.20   --cem-min-std-fraction 0.02   > artifacts/student_vehicle_sysid/fwd_v1_staged_cem_overnight_02.log 2>&1 &`


### Sys-id teaching data generation 
`python -m src.physx_teacher_dataset_builder \
  --dataset-dir artifacts/physx_teacher_datasets/comprehensive_fwd_v1 \
  --suite sysid-comprehensive-fwd \
  --generate-only`


#### SysID data recording 

`python -m src.physx_teacher_dataset_builder \
  --dataset-dir artifacts/physx_teacher_datasets/comprehensive_fwd_v1 \
  --suite sysid-comprehensive-fwd \
  --headless \
  --skip-existing \
  --skip-replays \
  --record-python /home/yz8733/miniforge3/envs/isaac-pytorch/bin/python`
