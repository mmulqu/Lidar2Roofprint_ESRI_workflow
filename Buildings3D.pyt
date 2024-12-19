import arcpy
from building_processor_tool import BuildingProcessor

class Toolbox(object):
    def __init__(self):
        """Define the toolbox"""
        self.label = "Building 3D Tools"
        self.alias = "buildings3d"
        self.tools = [BuildingProcessor]