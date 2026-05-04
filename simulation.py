import os, sys, argparse, warnings
import numpy as np
import nibabel as nib
import pyvista as pv
from scipy.ndimage import laplace, gaussian_filter

warnings.filterwarnings("ignore")

class IntegratedTumorSim:
    def __init__(self, npz_path):
        # 1. Chargement des données préparées (.npz)
        # If a directory is provided, construct the path to the NPZ file
        if os.path.isdir(npz_path):
            patient_id = os.path.basename(npz_path)
            npz_path = os.path.join(npz_path, f"{patient_id}_sim_data.npz")
        
        self.data = np.load(npz_path, allow_pickle=True)
        self.u = self.data['u0'].astype(np.float32)          # Densité initiale
        self.mask = self.data['brain_mask'].astype(np.uint8)  # Domaine
        self.D_field = self.data['D_field'].astype(np.float32)# Carte de diffusion
        self.spacing = self.data['spacing'].astype(np.float32)
        self.rho = float(self.data['rho'])
        self.kill_rate = float(self.data['kill_rate'])
        
        self.dt = 0.1
        self.current_time = 0
        self.patient_id = os.path.basename(npz_path).split('_')[0]

        # 2. Setup du Plotter Esthétique
        pv.set_plot_theme("dark")
        self.plotter = pv.Plotter(title=f"NEURO-SIM 3D : {self.patient_id}", window_size=(1400, 900))
        self.plotter.set_background("#010208")

        # Grille de base
        self.grid = pv.ImageData()
        self.grid.dimensions = np.array(self.u.shape)
        self.grid.spacing = self.spacing

        # 3. Rendu du Cerveau (Style Holoscan)
        print("[Visualisation] Création du maillage cérébral...")
        brain_sm = gaussian_filter(self.mask.astype(np.float32), sigma=1.5)
        self.brain_mesh = self.grid.contour([0.5], scalars=brain_sm.flatten(order="F"))
        self.plotter.add_mesh(self.brain_mesh, color="#66ccff", opacity=0.08, label="Cerveau (Anatomie)")
        self.plotter.add_mesh(self.brain_mesh, color="#3399ff", opacity=0.04, style="wireframe", line_width=0.5)

        # 4. Rendu de la Tumeur (Style PBR / Glow)
        self.grid.point_data["densite"] = self.u.flatten(order="F")
        self.tumeur_mesh = self.grid.contour([0.2], scalars="densite")
        
        # Coeur de la tumeur avec rendu physique (PBR)
        self.tumeur_actor = self.plotter.add_mesh(
            self.tumeur_mesh, color="#ff1100", opacity=1.0, 
            pbr=True, metallic=0.3, roughness=0.2, ambient=0.3, label="Masse Cellulaire"
        )
        # Halo (Glow)
        self.glow_actor = self.plotter.add_mesh(
            self.tumeur_mesh.copy(), color="#ff6600", opacity=0.2, smooth_shading=True
        )

        # 5. Éclairage et HUD
        self.setup_lighting()
        self.plotter.add_legend()
        self.plotter.enable_eye_dome_lighting() # Effet de relief

    def setup_lighting(self):
        self.plotter.remove_all_lights()
        self.plotter.add_light(pv.Light(position=(500, 500, 500), intensity=0.9, color="white"))
        self.plotter.add_light(pv.Light(position=(-500, -500, 0), intensity=0.5, color="#2244ff"))
        self.plotter.add_light(pv.Light(position=(0, 0, -500), intensity=0.4, color="red"))

    def solve_step(self):
        """Logique Fisher-KPP : Diffusion + Réaction"""
        # Diffusion hétérogène (D varie selon le voxel)
        lap = laplace(self.u) / (self.spacing[0]**2)
        
        # du/dt = D*lap + rho*u*(1-u) - kill*u
        diffusion = self.D_field * lap
        proliferation = self.rho * self.u * (1.0 - self.u)
        traitement = self.kill_rate * self.u
        
        self.u += self.dt * (diffusion + proliferation - traitement)
        
        # Contraintes
        self.u *= self.mask
        np.clip(self.u, 0, 1, out=self.u)
        self.current_time += self.dt

    def run(self, days=50):
        self.plotter.show(interactive_update=True)
        steps = int(days / self.dt)

        for i in range(steps):
            self.solve_step()

            # Mise à jour graphique différée (tous les 10 pas = 1 jour)
            if i % 10 == 0:
                self.grid.point_data["densite"] = self.u.flatten(order="F")
                
                # Mise à jour des maillages (Tumeur + Glow)
                try:
                    new_mesh = self.grid.contour([0.2], scalars="densite")
                    self.tumeur_actor.mapper.dataset.copy_from(new_mesh)
                    self.glow_actor.mapper.dataset.copy_from(new_mesh)
                except:
                    pass

                # Mise à jour HUD
                vol = np.sum(self.u > 0.2) * np.prod(self.spacing) / 1000
                hud_text = (f"PATIENT: {self.patient_id}\nJOUR: {self.current_time:.1f}\n"
                            f"VOLUME: {vol:.2f} cm3")
                self.plotter.add_text(hud_text, name="hud", position="upper_left", 
                                      font_size=10, font="courier", color="#00ffcc")
                
                self.plotter.update()

        print("Simulation terminée.")
        self.plotter.show()

if __name__ == "__main__":
    # Utilisation : python sim_complete.py P01_sim_data.npz
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Le fichier .npz préparé")
    args = parser.parse_args()

    sim = IntegratedTumorSim(args.file)
    sim.run(days=100) # Lance la simulation sur 100 jours