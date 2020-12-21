from pathlib import Path

import utils

import tools.via

if __name__ == "__main__":
  path = (
      r'D:\repo\ThermalImage\AnomalyDetection\gongrung1&2_2020-11-26\ir\via_project_27Nov2020_9h7m (3).json'
  )
  save_dir = Path(
      r'D:\repo\ThermalImage\AnomalyDetection\gongrung1&2_2020-11-26\mask')

  if not save_dir.exists():
    save_dir.mkdir()

  prj = tools.via.VIAProject(path=path,
                             attribute_name='thermal',
                             attributes_ids=['wall', 'other'])

  for file in prj.files:
    prj.write_masks(fname=file, save_dir=save_dir, shape=(240, 320))
