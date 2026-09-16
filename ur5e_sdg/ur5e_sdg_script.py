from omni.isaac.kit import SimulationApp
import os
import argparse

parser = argparse.ArgumentParser("Dataset generator")
parser.add_argument("--headless", type=bool, default=False, help="Launch script headless, default is False")
parser.add_argument("--height", type=int, default=544, help="Height of image")
parser.add_argument("--width", type=int, default=960, help="Width of image")
parser.add_argument("--num_frames", type=int, default=1000, help="Number of frames to record")
parser.add_argument("--data_dir", type=str, default=os.getcwd() + "/training_data", 
                    help="Location where data will be output")

args, unknown_args = parser.parse_known_args()

# This is the config used to launch simulation. 
CONFIG = {"renderer": "RayTracedLighting", "headless": args.headless, 
          "width": args.width, "height": args.height, "num_frames": args.num_frames}

simulation_app = SimulationApp(launch_config=CONFIG)

## This is the path which has the background scene in which objects will be added.
ENV_URL = os.path.join(os.path.dirname(__file__), "models/table_setup.usd")
TRAY_URL = os.path.join(os.path.dirname(__file__), "models/tray.usd")

import carb
import omni
import omni.usd
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import get_current_stage, open_stage
from pxr import Semantics
import omni.replicator.core as rep
from omni.isaac.core.utils.semantics import get_semantics

# Increase subframes if shadows/ghosting appears of moving objects
rep.settings.carb_settings("/omni/replicator/RTSubframes", 25)

CUBES = [
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/Shapes/cube.usd"
]

CYLINDERS = [
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/Shapes/cylinder.usd"
]

# This will handle replicator
def run_orchestrator():

    rep.orchestrator.run()

    # Wait until started
    while not rep.orchestrator.get_is_started():
        simulation_app.update()

    # Wait until stopped
    while rep.orchestrator.get_is_started():
        simulation_app.update()

    rep.BackendDispatch.wait_until_done()
    rep.orchestrator.stop()


def main():
    # Open the environment in a new stage
    print(f"Loading Stage {ENV_URL}")
    open_stage(ENV_URL)
    stage = get_current_stage()

    # Run some app updates to make sure things are properly loaded
    for i in range(100):
        if i % 10 == 0:
            print(f"App update {i}..")
        simulation_app.update()

    # Create camera with Replicator API for gathering data
    cam = rep.create.camera(focal_length=1.93, focus_distance=0.8, horizontal_aperture=3.896, clipping_range=(0.1, 1000000))

    # Create a tray
    tray = rep.create.from_usd(TRAY_URL)

    # Plane for scattering objects
    plane = rep.create.plane(scale=(0.35, 0.5, 1), visible=False, position=(0.6, 0, 1.12))

    # trigger replicator pipeline
    with rep.trigger.on_frame(num_frames=CONFIG["num_frames"]):

        # Randomize ground color
        with rep.get.prims(path_pattern="GroundPlane"):
            rep.randomizer.color(colors=rep.distribution.uniform(0.15, 0.5))
        
        # Randomize lighting
        with rep.get.prims(path_pattern="DomeLight"):
            rep.modify.attribute("color", rep.distribution.uniform((0.9, 0.9, 0.9), (1, 1, 1)))
            rep.modify.attribute("intensity", rep.distribution.uniform(600, 1200))

        # Randomize camera properties
        with cam:
            rep.modify.pose(
                position=rep.distribution.uniform((1, 0, 1.77), (1, 0, 1.83)),
                rotation=rep.distribution.uniform((0, -52, 0), (0, -48, 0))) # ZYX rotation where frames rotate with the sequence!
            
        # Randomize tray and plane pose
        random_position = rep.distribution.uniform((0.57, -0.03, 1.095), (0.63, 0.03, 1.095))
        random_rotation = rep.distribution.uniform((0, 0, -2), (0, 0, 2))
        with tray:
            rep.modify.pose(
                position=random_position,
                rotation=random_rotation)
        with plane:
            rep.modify.pose(
                position=random_position,
                rotation=random_rotation)

        # Randomize object properties
        with rep.randomizer.instantiate(paths=[CUBES[0]], 
                                        size=rep.distribution.uniform(0, 5),
                                        semantics=[("class", "cube")]):
            rep.modify.pose(rotation=rep.distribution.uniform((0, 0, 0), (45, 45, 180)),
                            scale=rep.distribution.uniform((0.03, 0.03, 0.03), (0.06, 0.06, 0.06)))
            rep.randomizer.scatter_2d(surface_prims=plane, check_for_collisions=False)
            rep.randomizer.color(colors=rep.distribution.uniform((0.6, 0.0, 0.6), (1.0, 0.3, 1.0)))

        with rep.randomizer.instantiate(paths=[CYLINDERS[0]], 
                                        size=rep.distribution.uniform(0, 5),
                                        semantics=[("class", "cylinder")]):
            rep.modify.pose(rotation=rep.distribution.uniform((0, 0, 0), (45, 45, 180)),
                            scale=rep.distribution.uniform((0.04, 0.04, 0.03), (0.06, 0.06, 0.06)))
            rep.randomizer.scatter_2d(surface_prims=plane, check_for_collisions=False)
            rep.randomizer.color(colors=rep.distribution.uniform((0.0, 0.0, 0.0), (0.3, 0.3, 0.3)))


    # Set up the writer
    writer = rep.WriterRegistry.get("KittiWriter")

    # output directory of writer
    output_directory = args.data_dir
    print("Outputting data to ", output_directory)

    # use writer for bounding boxes, rgb and segmentation
    writer.initialize(
        output_dir=output_directory,
        omit_semantic_type=True
    )

    # attach camera render products to wrieter so that data is outputted
    RESOLUTION = (CONFIG["width"], CONFIG["height"])
    render_product  = rep.create.render_product(cam, RESOLUTION)
    writer.attach(render_product)

    # run rep pipeline
    run_orchestrator()
    simulation_app.update()



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        carb.log_error(f"Exception: {e}")
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
