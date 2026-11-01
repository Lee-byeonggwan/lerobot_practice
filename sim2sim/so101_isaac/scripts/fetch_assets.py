"""Fetch public upstream assets and verify the exact bytes used by this experiment."""
import hashlib
from pathlib import Path
import shutil
import urllib.request
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
ASSETS = (
    ('https://github.com/LightwheelAI/leisaac/releases/download/v0.1.2/table_with_cube.zip',
     'downloads/table_with_cube.zip', '917c66a724019d235cc9f442a30ae72e5663b44ef4ed8d4d5324e549e11952b7'),
    ('https://huggingface.co/LightwheelAI/leisaac_env/resolve/main/assets/robots/so101_follower.usd',
     'assets/robots/so101_follower.usd', '64a877c3b82cdc4a48ab8a1f321a2dd3ef7c55d4b10bce222b58c530d978ae58'),
)

def main():
    for url, name, expected in ASSETS:
        path = ROOT/name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temp = path.with_suffix(path.suffix+'.part')
            with urllib.request.urlopen(url, timeout=120) as response, temp.open('wb') as output:
                shutil.copyfileobj(response, output)
            with temp.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != expected:
                raise ValueError(f'Upstream asset hash changed: {name}; partial download retained for inspection')
            temp.replace(path)
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, name
        print('SHA256 verified:', name)
    target = ROOT/'assets/scenes'
    if (target/'table_with_cube').exists():
        raise FileExistsError('Scene already exists; no extraction performed')
    target.mkdir(parents=True, exist_ok=True)
    with ZipFile(ROOT/'downloads/table_with_cube.zip') as archive:
        for item in archive.infolist():
            if not (target/item.filename).resolve().is_relative_to(target.resolve()):
                raise ValueError('Unexpected archive path')
        archive.extractall(target)
    assert (target/'table_with_cube/scene.usd').is_file()

if __name__ == '__main__':
    main()
