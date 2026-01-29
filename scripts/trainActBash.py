'''
!lerobot-train \
  --dataset.repo_id=dariusss04/multicolor-cube \
  --dataset.root=/kaggle/working/multicolor-cube \
  --policy.type=act \
  --policy.use_vae=true \
  --policy.use_amp=true \
  --batch_size=8 \
  --steps=60000 \
  --log_freq=50 \
  --save_checkpoint=true \
  --save_freq=2000 \
  --output_dir=/kaggle/working/outputs/act_multicolor_lerobot \
  --job_name=act_multicolor_lerobot \
  --policy.push_to_hub=true \
  --policy.repo_id=dariusss04/act-leRobot-multiColorCube \
  --resume=false
  '''