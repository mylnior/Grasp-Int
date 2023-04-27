import depthai as dai

class DepthMapper:
    def __init__(self) -> None:
        self.create_pipeline()

    def create_pipeline(self):
        print("Creating pipeline...")
        # Start defining a pipeline
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_4)


        # ColorCamera
        print("Creating Color Camera...")
        camRgb = pipeline.createColorCamera()
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.resolution = (1920,1080)
        controlIn = pipeline.create(dai.node.XLinkIn)
        controlIn.setStreamName('control')
        controlIn.out.link(camRgb.inputControl)

        camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        camRgb.setInterleaved(False)
        camRgb.setIspScale(2, 3)
        lensPos = 150
        self.expTime = 8000
        self.sensIso = 500    
        self.wbManual = 4000
        print("Setting manual exposure, time: ", self.expTime, "iso: ", self.sensIso)
        camRgb.initialControl.setManualExposure(self.expTime, self.sensIso)
        # cam.initialControl.setAutoExposureEnable()
        camRgb.initialControl.setManualWhiteBalance(self.wbManual)
        # cam.initialControl.setManualFocus(lensPos)
        # cam.setIspScale(self.scale_nd[0], self.scale_nd[1])
        camRgb.setFps(self.fps)

        camLeft = pipeline.create(dai.node.MonoCamera)
        camRight = pipeline.create(dai.node.MonoCamera)
        camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        res = dai.MonoCameraProperties.SensorResolution.THE_720_P
        # cam.setVideoSize(self.resolution)
        for monoCam in (camLeft, camRight):  # Common config
            monoCam.setResolution(res)
            monoCam.setFps(self.fps)
        self.left_right_initial_res = 720

        self.resolutions = [    (1920, 1080),    (1280, 720),    (854, 480),    (640, 360),    (426, 240)]
        self.res_idx = 1


        cam_out = pipeline.createXLinkOut()
        cam_out.setStreamName("rgb")
        cam_out.input.setQueueSize(1)
        cam_out.input.setBlocking(False)
        camRgb.isp.link(cam_out.input)

        self.landmark_depth=False
        self.depth_map_depth = True

        streams = ["rgb"]
        self.hands['rgb']=[]
        extracts={}
        extracts['rgb'] = self.extract_hands_rgb
        # Create StereoDepth node that will produce the depth map
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.initialConfig.setConfidenceThreshold(245)
        stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
        stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        camLeft.out.link(stereo.left)
        camRight.out.link(stereo.right)

        # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
        extended_disparity = True
        # Better accuracy for longer distance, fractional disparity 32-levels:
        subpixel = True
        # Better handling for occlusions:
        lr_check = True

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        stereo.setLeftRightCheck(lr_check)
        stereo.setExtendedDisparity(extended_disparity)
        stereo.setSubpixel(subpixel)
        depth_out = pipeline.create(dai.node.XLinkOut)
        depth_out.setStreamName("depth")
        stereo.depth.link(depth_out.input)
        print("Pipeline created.")
        return pipeline