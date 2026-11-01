"""Show existing sensor RGB buffers together, without extra camera rendering."""
import asyncio
import time
from pathlib import Path
import numpy as np
import omni.ui as ui
import omni.kit.app


class LiveCameraPanel:
    def __init__(self, env, reference_dir=None):
        self.env = env
        self.names = ('top', 'wrist', 'front')
        self.providers = {name: ui.ByteImageProvider() for name in self.names}
        self.buffers = {}
        self.reference_providers = {}
        self.reference_buffers = {}
        if reference_dir is not None:
            from PIL import Image
            for name in self.names:
                role = 'belly' if name == 'front' else name
                with Image.open(Path(reference_dir)/(role+'.png')) as image:
                    rgba = np.array(image.convert('RGBA'))
                provider = ui.ByteImageProvider()
                self.reference_buffers[name] = rgba
                provider.set_bytes_data(rgba.flatten().data, [rgba.shape[1], rgba.shape[0]])
                self.reference_providers[name] = provider
        self.last_update = 0.
        self.updates = 0
        self.window = ui.Window('Real / Sim Camera Comparison' if reference_dir else 'Live Cameras - Top / Wrist / Belly',
                                width=760 if reference_dir else 410, height=740)
        with self.window.frame:
            with ui.VStack(spacing=3):
                self.status = ui.Label('Three live sensor views | preview up to 10 Hz', height=20)
                if self.reference_providers:
                    ui.Label('LEFT: saved REAL photo | RIGHT: live SIM | poses not yet matched',height=20)
                for name in self.names:
                    with ui.VStack(height=ui.Fraction(1), spacing=1):
                        ui.Label('Belly' if name == 'front' else name.title(), height=18)
                        with ui.HStack():
                            if self.reference_providers:
                                ui.ImageWithProvider(self.reference_providers[name],
                                                     fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                            ui.ImageWithProvider(self.providers[name],
                                                 fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
        self.dock_task = asyncio.ensure_future(self._dock())

    async def _dock(self):
        for _ in range(3):
            await omni.kit.app.get_app().next_update_async()
        viewport = ui.Workspace.get_window('Viewport')
        if viewport is not None:
            self.window.dock_in(viewport, ui.DockPosition.RIGHT, .55 if self.reference_providers else .32)

    def show(self):
        self.window.visible = True

    def update(self, force=False):
        now = time.monotonic()
        if not force and (not self.window.visible or now - self.last_update < .1):
            return
        for name in self.names:
            rgb = self.env.scene[name].data.output['rgb'].torch[0, :, :, :3].detach().cpu().numpy()
            rgba = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
            rgba[:, :, :3] = rgb
            rgba[:, :, 3] = 255
            self.buffers[name] = rgba
            self.providers[name].set_bytes_data(rgba.flatten().data, [rgba.shape[1], rgba.shape[0]])
        self.last_update = now
        self.updates += 1
        self.status.text = f'Top / Wrist / Belly | live preview #{self.updates} | up to 10 Hz'
