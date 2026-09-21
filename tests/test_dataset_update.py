import importlib.util,json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('build_dataset',Path(__file__).parents[1]/'scripts/build_dataset.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def test_group_split_and_dedup(tmp_path):
    rows=[]
    for i in range(20):
        for language in ('en','ta','tanglish'):
            rows.append(dict(group_id=f'g{i}',language=language,source='test',license='CC0-1.0',text=f'Example {i} in {language}.',instruction=f'Question {i} in {language}?',input='',output=f'Answer {i}.'))
    rows.append(rows[0].copy())
    source=tmp_path/'input.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    out=tmp_path/'ready';report=mod.build(source,out,300)
    assert report['duplicates_removed']==1
    sets=[]
    for split in ('train','validation','test'):
        records=[json.loads(s) for s in (out/'sft'/f'{split}.jsonl').read_text().splitlines()]
        sets.append({r['group_id'] for r in records})
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    assert sum(report['records'].values())==60
    assert report['tokenizer_training_split']=='train only'
    with pytest.raises(ValueError,match='exists'):mod.build(source,out,300)

def test_real_starter_integrity():
    root=Path('data/ready-v1'); seen=set()
    for split in ('train','validation','test'):
        rows=[json.loads(x) for x in (root/'sft'/f'{split}.jsonl').read_text().splitlines()]
        groups={r['split_group'] for r in rows}
        assert not seen&groups;seen|=groups
        assert {r['language'] for r in rows}=={'en','ta','tanglish'}
        assert all(r['synthetic'] is True for r in rows)
