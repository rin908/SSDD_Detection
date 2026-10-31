"""纯合成小数据回归检查；不读写真实 SSDD，不运行模型。"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from prepare_data import prepare, pixel_hash


class PrepareDataTest(unittest.TestCase):
    def test_normalization_leakage_and_repeatability(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source, out = base / 'source', base / 'output'
            def write_split(split, entries):
                images, annotations = [], []
                (source / 'images' / split).mkdir(parents=True)
                (source / 'annotations').mkdir(exist_ok=True)
                for image_id, color in entries:
                    name = f'{image_id:06}.png'
                    Image.new('RGB', (16, 16), (color, color, color)).save(source / 'images' / split / name)
                    images.append(dict(id=image_id, file_name=name, width=16, height=16))
                    annotations.append(dict(id=image_id, image_id=image_id, bbox=[0, 0, 4, 5],
                                            area=7, category_id=0, iscrowd=0))
                doc = dict(images=images, annotations=annotations,
                           categories=[dict(id=0, name='ship')])
                (source / 'annotations' / f'{split}.json').write_text(json.dumps(doc), encoding='utf-8')
            write_split('train', [(i, i) for i in range(1, 11)])
            write_split('test', [(20, 1), (21, 21)])
            write_split('test_inshore', [(20, 1)])
            write_split('test_offshore', [(21, 21)])
            source_before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):
                report = prepare(source, out)
            self.assertEqual(report['main_counts'], dict(train=8, val=1, test=2))
            self.assertEqual(report['removed_train'], 1)
            docs = {split: json.loads((out / 'annotations' / f'{split}_norm.json').read_text())
                    for split in ('train', 'val', 'test')}
            for doc in docs.values():
                self.assertEqual(doc['categories'][0]['id'], 1)
                for ann in doc['annotations']:
                    self.assertEqual((ann['category_id'], ann['area']), (1, 20))
            hashes = {split: {pixel_hash(out / 'images' / im['file_name']) for im in doc['images']}
                      for split, doc in docs.items()}
            self.assertFalse(hashes['train'] & hashes['test'])
            self.assertFalse(hashes['train'] & hashes['val'])
            self.assertFalse(hashes['val'] & hashes['test'])
            before = (out / 'annotations/train_norm.json').read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(source, out)
            self.assertEqual(before, (out / 'annotations/train_norm.json').read_bytes())
            self.assertEqual(source_before, {str(p.relative_to(source)): p.read_bytes()
                                            for p in source.rglob('*') if p.is_file()})
            original = json.loads((out / 'annotations/original/train_norm.json').read_text())
            self.assertEqual(len(original['images']), 9)


if __name__ == '__main__':
    unittest.main()
