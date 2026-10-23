import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/"experiments"))
SPEC = importlib.util.spec_from_file_location("reconstruct", PROJECT/"experiments/reconstruct.py")
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


class ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (PROJECT/"results").mkdir(exist_ok=True)

    def test_preset_and_matching_recipe(self):
        preset = r.load_preset(PROJECT/"configs/rosmontis_71.json")
        self.assertEqual(list(map(len,preset["groups"].values())),[24,23,24])
        commands = dict(r.sparse_commands(Path("runs/test"),preset))
        extraction = commands["feature_extractor"]
        self.assertEqual(extraction[extraction.index("--SiftExtraction.peak_threshold")+1],"0.003")
        self.assertIn("--GlobalMapper.ba_refine_extra_params",commands["global_mapper"])
        self.assertNotIn("--FeatureMatching.guided_matching",commands["exhaustive_matcher"])

    def test_ambiguous_source_names_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=PROJECT/"results",prefix="reconstruct-test-") as name:
            root = Path(name)
            (root/"other").mkdir()
            for path in (root/"frame.JPG",root/"other/frame.JPG"):
                Image.new("RGB",(2,2)).save(path)
            with self.assertRaises(ValueError):
                r.locate_images(root,dict(groups={"low":["frame.JPG"]},raw_size=[2,2]))

    def test_output_guards_prevent_reuse_and_source_overlap(self):
        with tempfile.TemporaryDirectory(dir=PROJECT/"results",prefix="reconstruct-test-") as name:
            root = Path(name)
            source = root/"data"
            source.mkdir()
            existing = root/"runs/existing"
            existing.mkdir(parents=True)
            with patch.object(r,"PROJECT",root):
                with self.assertRaises(FileExistsError):
                    r.safe_output(existing,source)
                with self.assertRaises(ValueError):
                    r.safe_output(root/"docs/new",source)
                with self.assertRaises(ValueError):
                    r.safe_output(existing/"child",existing)

    def test_view_context_rebuilds_from_new_poses_without_old_results(self):
        preset = r.load_preset(PROJECT/"configs/rosmontis_71.json")
        images, matrices = {}, {}
        with tempfile.TemporaryDirectory(dir=PROJECT/"results",prefix="reconstruct-test-") as name:
            run = Path(name)
            (run/"images").mkdir()
            for group_index,names in enumerate(preset["groups"].values()):
                for i,source in enumerate(names):
                    angle = 2*np.pi*i/len(names)
                    center = np.array([2*np.cos(angle),2*np.sin(angle),group_index*.6])
                    forward = -center/np.linalg.norm(center)
                    right = np.cross(forward,[0.,0.,1.]); right /= np.linalg.norm(right)
                    rotation = np.array([right,np.cross(forward,right),forward])
                    key = len(images)
                    matrices[key] = rotation
                    image_name = Path(source).with_suffix(".PNG").name
                    images[key] = dict(name=image_name,q=key,t=-rotation@center)
                    Image.new("RGB",(12,8),(100+i,120,140)).save(run/"images"/image_name)
            points = {i:dict(xyz=np.array(x)) for i,x in enumerate([
                (-.2,-.2,-.3),(.2,-.2,-.3),(-.2,.2,-.3),(.2,.2,-.3),
                (-.2,-.2,.3),(.2,-.2,.3),(-.2,.2,.3),(.2,.2,.3)])}
            with patch.object(r,"read_model",return_value=({},images,points)),patch.object(r,"rotation",side_effect=lambda key:matrices[key]):
                r.prepare_context(run,preset)
            saved = json.loads((run/"render_context.json").read_text(encoding="utf-8"))
            transform = np.array(saved["metadata"]["cloud_view"]["model_to_upright"])
            np.testing.assert_allclose(transform@transform.T,np.eye(3),atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(transform),1.)
            self.assertEqual(saved["source_model"],"final/sparse/0")
            self.assertEqual(len(saved["metadata"]["frame_sequence"]),24)
            with Image.open(run/"photo_source.gif") as gif:
                self.assertEqual(gif.size,(903,450))
                self.assertEqual(gif.n_frames,24)


if __name__ == "__main__":
    unittest.main()
