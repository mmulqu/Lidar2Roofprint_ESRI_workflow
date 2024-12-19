import arcpy
import os
import sys
import time
import re
from arcpy.sa import *


def check_building_class_code(las_dataset_path):
    """
    Check if the LAS dataset is valid and contains points.
    """
    try:
        desc = arcpy.Describe(las_dataset_path)

        if desc.dataType != "LasDataset":
            arcpy.AddError(f"Input is not a valid LAS dataset: {las_dataset_path}")
            return False

        arcpy.AddMessage(f"LAS Dataset information:")
        arcpy.AddMessage(f"  - Point count: {desc.pointCount}")
        arcpy.AddMessage(f"  - File count: {desc.fileCount}")

        return True

    except Exception as e:
        arcpy.AddWarning(f"Error checking LAS dataset: {str(e)}")
        arcpy.AddMessage("Continuing processing despite validation check failure...")
        return True


def validate_roof_forms(project_ws):
    """
    Check for roof forms output using multiple possible names
    """
    possible_names = [
        os.path.join(project_ws, "roof_forms"),
        os.path.join(project_ws, "roof_forms_roofform")
    ]
    for output_path in possible_names:
        if arcpy.Exists(output_path):
            arcpy.AddMessage(f"Found roof forms output at: {output_path}")
            return True
    return False


def process_las_dataset(las_dataset_path, home_directory, output_directory):
    """
    Process a single LAS dataset to extract building footprints and roof forms.
    """
    try:
        # Enable overwriting of outputs
        arcpy.env.overwriteOutput = True

        # Validate input paths
        if not arcpy.Exists(las_dataset_path):
            arcpy.AddError(f"LAS dataset does not exist: {las_dataset_path}")
            return False

        if not os.path.exists(home_directory):
            arcpy.AddError(f"Home directory does not exist: {home_directory}")
            return False

        if not os.path.exists(output_directory):
            arcpy.AddError(f"Output directory does not exist: {output_directory}")
            return False

        # Create or get the intermediate workspace
        intermediate_ws = os.path.join(output_directory, "Intermediate.gdb")
        if not arcpy.Exists(intermediate_ws):
            arcpy.CreateFileGDB_management(output_directory, "Intermediate")

        # Get LAS dataset name for unique naming
        lasd_name = os.path.splitext(os.path.basename(las_dataset_path))[0]

        # Create geodatabase for final outputs
        gdb_name = f"{lasd_name}_processing.gdb"
        project_ws = os.path.join(output_directory, gdb_name)

        if not arcpy.Exists(project_ws):
            arcpy.AddMessage(f"Creating new File Geodatabase: {project_ws}")
            arcpy.CreateFileGDB_management(output_directory, gdb_name)

        # Set workspace environments
        arcpy.env.workspace = project_ws
        arcpy.env.scratchWorkspace = intermediate_ws

        # Check LAS dataset validity
        if not check_building_class_code(las_dataset_path):
            arcpy.AddError(f"Invalid LAS dataset: {las_dataset_path}")
            return False

        # Create geodatabase
        lasd_name = os.path.splitext(os.path.basename(las_dataset_path))[0]
        gdb_name = f"{lasd_name}_processing.gdb"
        project_ws = os.path.join(output_directory, gdb_name)

        if not arcpy.Exists(project_ws):
            arcpy.AddMessage(f"Creating new File Geodatabase: {project_ws}")
            arcpy.CreateFileGDB_management(output_directory, gdb_name)

        arcpy.env.workspace = project_ws

        # ---------------------------------------------------------------------------
        # STEP 1: Extract Elevation from LAS Dataset
        # ---------------------------------------------------------------------------
        try:
            from scripts.extract_elevation_from_las import run as run_extract_elevation

            output_elevation_raster_base = os.path.join(project_ws, "elev")
            run_extract_elevation(
                home_directory=home_directory,
                project_ws=project_ws,
                input_las_dataset=las_dataset_path,
                cell_size="0.3",
                only_ground_plus_class_code=True,
                class_code=6,
                output_elevation_raster=output_elevation_raster_base,
                classify_noise=False,
                minimum_height="0.5",
                maximum_height="50",
                processing_extent="#",
                debug=1
            )

            # Validate elevation outputs
            for suffix in ['_dtm', '_dsm', '_ndsm']:
                raster_path = output_elevation_raster_base + suffix
                if not arcpy.Exists(raster_path):
                    raise ValueError(f"Elevation raster not created: {raster_path}")

        except Exception as e:
            arcpy.AddError(f"Error in elevation extraction: {str(e)}")
            return False

        # ---------------------------------------------------------------------------
        # STEP 2: Create Draft Footprint Raster
        # ---------------------------------------------------------------------------
        try:
            from scripts.create_building_mosaic import run as run_building_mosaic

            las_spatial_ref = arcpy.Describe(las_dataset_path).spatialReference
            if not las_spatial_ref:
                raise ValueError("Could not obtain spatial reference from LAS dataset")

            arcpy.AddMessage(f"Using spatial reference from LAS dataset: {las_spatial_ref.name}")

            # Set environment settings
            arcpy.env.cellSize = "0.6 Meters"
            arcpy.env.outputCoordinateSystem = las_spatial_ref

            mosaic_folder = os.path.join(output_directory, f"{lasd_name}_mosaic_rasters")
            if not os.path.exists(mosaic_folder):
                os.makedirs(mosaic_folder)

            out_mosaic = os.path.join(project_ws, "building_mosaic")

            run_building_mosaic(
                home_directory=home_directory,
                project_ws=project_ws,
                in_lasd=las_dataset_path,
                out_folder=mosaic_folder,
                out_mosaic=out_mosaic,
                spatial_ref=las_spatial_ref,
                cell_size="0.6 Meters",
                debug=1
            )

            if not arcpy.Exists(out_mosaic):
                raise ValueError("Building mosaic not created")

        except Exception as e:
            arcpy.AddError(f"Error in building mosaic creation: {str(e)}")
            return False

        # ---------------------------------------------------------------------------
        # STEP 3: Run Focal Statistics
        # ---------------------------------------------------------------------------
        try:
            arcpy.AddMessage("Running Focal Statistics...")
            focal_input = out_mosaic
            out_focal = os.path.join(project_ws, "focal_mosaic")

            if not arcpy.Exists(focal_input):
                raise ValueError(f"Input raster for focal statistics does not exist: {focal_input}")

            focal_result = FocalStatistics(
                in_raster=focal_input,
                neighborhood=NbrRectangle(3, 3, "CELL"),
                statistics_type="MAJORITY",
                ignore_nodata="DATA"
            )

            focal_result.save(out_focal)

            if not arcpy.Exists(out_focal):
                raise ValueError("Focal statistics output not created")

            arcpy.AddMessage("Focal Statistics completed successfully")

        except Exception as e:
            arcpy.AddError(f"Error in Focal Statistics: {str(e)}")
            return False

        # ---------------------------------------------------------------------------
        # STEP 4: Create footprints from raster with safe shrink operation
        # ---------------------------------------------------------------------------
        try:
            from scripts.footprints_from_raster import run as run_footprints_from_raster

            arcpy.AddMessage("Creating footprints from raster...")

            in_focal_raster = os.path.join(project_ws, "focal_mosaic")
            output_poly = os.path.join(project_ws, "final_footprints")

            if not arcpy.Exists(in_focal_raster):
                raise ValueError(f"Input focal raster does not exist: {in_focal_raster}")

            run_footprints_from_raster(
                home_directory=home_directory,
                project_ws=project_ws,
                in_raster=in_focal_raster,
                min_area="32 SquareMeters",
                split_features="",
                simplify_tolerance="1.5 Meters",
                output_poly=output_poly,
                reg_circles=True,
                circle_min_area="4000 SquareFeet",
                min_compactness=0.85,
                circle_tolerance="10 Feet",
                lg_reg_method="ANY_ANGLE",
                lg_min_area="25000 SquareFeet",
                lg_tolerance="2 Feet",
                med_reg_method="RIGHT_ANGLES_AND_DIAGONALS",
                med_min_area="5000 SquareFeet",
                med_tolerance="4 Feet",
                sm_reg_method="RIGHT_ANGLES",
                sm_tolerance="4 Feet",
                debug=1
            )

            if not arcpy.Exists(output_poly):
                raise ValueError("Building footprints not created")

            arcpy.AddMessage("Building footprints created successfully")

        except Exception as e:
            arcpy.AddError(f"Error in footprints creation: {str(e)}")
            return False

        # ---------------------------------------------------------------------------
        # STEP 5: Segment Roofs
        # ---------------------------------------------------------------------------
        try:
            from scripts.roof_part_segmentation import run as run_segment_roof

            arcpy.AddMessage("Starting roof segmentation process...")

            roof_segments_folder = os.path.join(home_directory, "roof_forms")
            if not os.path.exists(roof_segments_folder):
                os.makedirs(roof_segments_folder)

            dsm_path = os.path.join(project_ws, "elev_dsm")
            roof_segments = os.path.join(project_ws, "roof_segments")

            if not arcpy.Exists(dsm_path):
                raise ValueError(f"DSM raster does not exist: {dsm_path}")

            run_segment_roof(
                home_directory=home_directory,
                project_ws=project_ws,
                features=output_poly,
                dsm=dsm_path,
                spectral_detail="12",
                spatial_detail="12",
                minimum_segment_size=555,
                regularization_tolerance="3",
                flat_only=False,
                min_slope=10,
                output_segments_ui=roof_segments,
                debug=1
            )

            segmented_output = roof_segments + "_segmented"
            if not arcpy.Exists(segmented_output):
                raise ValueError("Roof segmentation failed")

            arcpy.AddMessage("Roof segmentation completed successfully")

        except Exception as e:
            arcpy.AddError(f"Error in roof segmentation: {str(e)}")
            return False

        # ---------------------------------------------------------------------------
        # STEP 6: Extract Roof Forms
        # ---------------------------------------------------------------------------
        try:
            from scripts.extract_roof_form import run as run_extract_roof_form

            arcpy.AddMessage("Starting roof form extraction...")

            segmented_roofs = os.path.join(project_ws, "roof_segments_segmented")
            dsm_path = os.path.join(project_ws, "elev_dsm")
            dtm_path = os.path.join(project_ws, "elev_dtm")
            ndsm_path = os.path.join(project_ws, "elev_ndsm")
            output_roofforms = os.path.join(project_ws, "roof_forms")

            if not arcpy.Exists(segmented_roofs):
                raise ValueError(f"Segmented roofs file not found: {segmented_roofs}")
            if not arcpy.Exists(dsm_path):
                raise ValueError(f"DSM file not found: {dsm_path}")
            if not arcpy.Exists(dtm_path):
                raise ValueError(f"DTM file not found: {dtm_path}")
            if not arcpy.Exists(ndsm_path):
                raise ValueError(f"nDSM file not found: {ndsm_path}")

            run_extract_roof_form(
                home_directory=home_directory,
                project_ws=project_ws,
                buildings_layer=segmented_roofs,
                dsm=dsm_path,
                dtm=dtm_path,
                ndsm=ndsm_path,
                flat_roofs=False,
                min_flat_roof_area="23",
                min_slope_roof_area="7",
                min_roof_height="2.5",
                output_buildings=output_roofforms,
                simplify_buildings="true",
                simplify_tolerance="0.3",
                debug=1
            )

            if not validate_roof_forms(project_ws):
                raise ValueError("Roof form extraction failed")

            arcpy.AddMessage("Roof form extraction completed successfully")
            return True

        except Exception as e:
            arcpy.AddError(f"Error in roof form extraction: {str(e)}")
            return False


    except arcpy.ExecuteError:
        arcpy.AddError("ArcPy error in process:")
        arcpy.AddError(arcpy.GetMessages(2))
        return False
    except Exception as e:
        arcpy.AddError(f"An unexpected error occurred: {str(e)}")
        return False


class BuildingProcessor(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "3D Building Processor"
        self.description = "Process LAS datasets to create 3D building footprints and roof forms"
        self.canRunInBackground = True

    def getParameterInfo(self):
        """Define parameter definitions"""
        # Input LAS Dataset
        param0 = arcpy.Parameter(
            displayName="Input LAS Dataset",
            name="las_dataset",
            datatype="DELasDataset",
            parameterType="Required",
            direction="Input")

        # Home Directory (containing scripts)
        param1 = arcpy.Parameter(
            displayName="Home Directory",
            name="home_directory",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input")
        param1.description = "Directory containing processing scripts"

        # Output Directory
        param2 = arcpy.Parameter(
            displayName="Output Directory",
            name="output_directory",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input")
        param2.description = "Directory where results will be stored"

        # Optional Parameters
        param3 = arcpy.Parameter(
            displayName="Cell Size",
            name="cell_size",
            datatype="GPString",
            parameterType="Optional",
            direction="Input")
        param3.value = "0.3"
        param3.description = "Cell size for elevation rasters (in meters)"

        param4 = arcpy.Parameter(
            displayName="Building Class Code",
            name="class_code",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input")
        param4.value = 6
        param4.description = "LAS classification code for buildings"

        param5 = arcpy.Parameter(
            displayName="Minimum Building Height",
            name="min_height",
            datatype="GPString",
            parameterType="Optional",
            direction="Input")
        param5.value = "0.5"
        param5.description = "Minimum height threshold for buildings (in meters)"

        params = [param0, param1, param2, param3, param4, param5]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckOutExtension("Spatial")
            else:
                raise arcpy.ExecuteError("Spatial Analyst license is unavailable")
        except:
            return False
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal validation."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool parameter."""
        # Validate LAS dataset
        if parameters[0].value:
            if not arcpy.Exists(parameters[0].valueAsText):
                parameters[0].setErrorMessage("LAS dataset does not exist")

        # Validate home directory
        if parameters[1].value:
            if not os.path.exists(parameters[1].valueAsText):
                parameters[1].setErrorMessage("Home directory does not exist")
            elif not os.path.exists(os.path.join(parameters[1].valueAsText, "scripts")):
                parameters[1].setWarningMessage("No 'scripts' folder found in home directory")

        # Validate output directory
        if parameters[2].value:
            if not os.path.exists(parameters[2].valueAsText):
                try:
                    os.makedirs(parameters[2].valueAsText)
                except:
                    parameters[2].setErrorMessage("Cannot create output directory")
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        try:
            # Get parameters
            las_dataset = parameters[0].valueAsText
            home_directory = parameters[1].valueAsText
            output_directory = parameters[2].valueAsText

            # Process the LAS dataset
            success = process_las_dataset(las_dataset, home_directory, output_directory)

            if success:
                arcpy.AddMessage("Processing completed successfully")
            else:
                arcpy.AddError("Processing failed - check messages above")

        except arcpy.ExecuteError:
            arcpy.AddError(arcpy.GetMessages(2))
        except Exception as e:
            arcpy.AddError(f"An unexpected error occurred: {str(e)}")
        finally:
            # Clean up
            try:
                arcpy.CheckInExtension("Spatial")
            except:
                pass