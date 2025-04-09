#!/usr/bin/env python3
import rospy
import numpy as np
import pycuda.autoinit  # Critical for CUDA context
import pycuda.driver as cuda
import tensorrt as trt
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class EdgeSLAMNode:
    def __init__(self):
        rospy.init_node('edge_slam')
        self.bridge = CvBridge()
        self.pub_depth = rospy.Publisher('/depth', Image, queue_size=10)
        
        # Load TensorRT engine
        self.engine = self.load_engine("depth_net.engine")
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        self._allocate_buffers()
        
        # Subscribe to RGB topic
        rospy.Subscriber('/camera/rgb', Image, self.image_callback)

    def load_engine(self, engine_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def preprocess(self, cv_image):
        # Convert to CHW, normalize, add batch dim
        image = cv_image.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.ascontiguousarray(image[np.newaxis, ...])

    def infer(self, image_np):
        # Copy input to GPU
        np.copyto(self.inputs[0]['host'], image_np.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]['device'], 
            self.inputs[0]['host'], 
            self.stream
        )
        
        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # Copy outputs back
        cuda.memcpy_dtoh_async(
            self.outputs[0]['host'], 
            self.outputs[0]['device'], 
            self.stream
        )
        cuda.memcpy_dtoh_async(
            self.outputs[1]['host'], 
            self.outputs[1]['device'], 
            self.stream
        )
        self.stream.synchronize()
        
        # Reshape outputs (batch, 1, H, W)
        depth_mean = self.outputs[0]['host'].reshape(1, 1, *image_np.shape[2:])
        depth_var = self.outputs[1]['host'].reshape(1, 1, *image_np.shape[2:])
        return depth_mean, depth_var

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            
            # Preprocess and infer
            image_np = self.preprocess(cv_image)
            depth_mean, _ = self.infer(image_np)
            
            # Publish depth
            depth_msg = self.bridge.cv2_to_imgmsg(
                depth_mean[0, 0].astype(np.float32), 
                encoding="32FC1"
            )
            self.pub_depth.publish(depth_msg)
        except Exception as e:
            rospy.logerr(f"Error in callback: {e}")

if __name__ == "__main__":
    node = EdgeSLAMNode()
    rospy.spin()