import React, { useEffect, useRef, useState, useCallback } from 'react';
import { MediaPipeManager } from '../services/mediapipeManager';
import { extractFrameFeatures } from '../services/featureExtraction';
import { FrameFeatures } from '../types/features';
import { FEATURE_CONFIG } from '../config/featureConfig';

interface WebcamFeedProps {
  isActive: boolean;
  onFrameFeatures?: (features: FrameFeatures) => void;
  showOverlay?: boolean;
}

const WebcamFeed: React.FC<WebcamFeedProps> = ({ 
  isActive, 
  onFrameFeatures,
  showOverlay = false 
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [faceDetected, setFaceDetected] = useState<boolean>(false);
  const [fps, setFps] = useState<number>(0);
  
  const mediaPipeRef = useRef<MediaPipeManager | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef<number>(0);
  const frameCountRef = useRef<number>(0);
  const fpsUpdateIntervalRef = useRef<number>(0);

  // Initialize MediaPipe
  useEffect(() => {
    const initMediaPipe = async () => {
      try {
        const mediaPipe = new MediaPipeManager();
        await mediaPipe.initialize();
        mediaPipeRef.current = mediaPipe;
        console.log('MediaPipe initialized');
      } catch (err) {
        console.error('Failed to initialize MediaPipe:', err);
        setError('Failed to initialize face tracking');
      }
    };

    initMediaPipe();

    return () => {
      if (mediaPipeRef.current) {
        mediaPipeRef.current.close();
        mediaPipeRef.current = null;
      }
    };
  }, []);

  // Process video frames
  const processFrame = useCallback(async () => {
    if (!isActive || !videoRef.current || !canvasRef.current || !mediaPipeRef.current) {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;

    // Check if video is ready
    if (video.readyState !== video.HAVE_ENOUGH_DATA) {
      animationFrameRef.current = requestAnimationFrame(processFrame);
      return;
    }

    try {
      setIsProcessing(true);

      // Process frame with MediaPipe
      await mediaPipeRef.current.processFrame(video);
      const landmarkResult = mediaPipeRef.current.getLastResult();

      // Update face detection status
      setFaceDetected(landmarkResult.detected);

      // Extract frame features
      const ctx = canvas.getContext('2d');
      if (ctx) {
        // Draw video frame to canvas
        canvas.width = video.videoWidth || FEATURE_CONFIG.video.width;
        canvas.height = video.videoHeight || FEATURE_CONFIG.video.height;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Get image data
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        // Extract features
        const frameFeatures = extractFrameFeatures(imageData, landmarkResult);

        // Call callback with features
        if (onFrameFeatures) {
          onFrameFeatures(frameFeatures);
        }
      }

      // Update FPS counter
      frameCountRef.current++;
      const now = performance.now();
      if (now - fpsUpdateIntervalRef.current >= 1000) {
        const elapsed = (now - fpsUpdateIntervalRef.current) / 1000;
        setFps(Math.round(frameCountRef.current / elapsed));
        frameCountRef.current = 0;
        fpsUpdateIntervalRef.current = now;
      }

      setIsProcessing(false);
    } catch (err) {
      console.error('Frame processing error:', err);
      setIsProcessing(false);
    }

    // Schedule next frame
    animationFrameRef.current = requestAnimationFrame(processFrame);
  }, [isActive, onFrameFeatures]);

  // Setup webcam
  useEffect(() => {
    if (isActive) {
      navigator.mediaDevices
        .getUserMedia({ 
          video: { 
            width: FEATURE_CONFIG.video.width,
            height: FEATURE_CONFIG.video.height,
            frameRate: FEATURE_CONFIG.video.fps
          } 
        })
        .then((mediaStream) => {
          setStream(mediaStream);
          if (videoRef.current) {
            videoRef.current.srcObject = mediaStream;
          }
        })
        .catch((err) => {
          console.error('Camera error:', err);
          setError('Unable to access camera');
        });
    } else {
      // Stop processing when inactive
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    }

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isActive]);

  // Start frame processing when video is ready
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleLoadedData = () => {
      console.log('Video ready, starting frame processing');
      frameCountRef.current = 0;
      fpsUpdateIntervalRef.current = performance.now();
      processFrame();
    };

    video.addEventListener('loadeddata', handleLoadedData);

    return () => {
      video.removeEventListener('loadeddata', handleLoadedData);
    };
  }, [processFrame]);

  if (error) {
    return (
      <div className="bg-gray-200 rounded-lg p-4 text-center text-sm text-gray-600">
        {error}
      </div>
    );
  }

  return (
    <div className="relative">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="rounded-lg shadow-md"
        style={{ width: '320px', height: '240px', display: showOverlay ? 'none' : 'block' }}
      />
      <canvas
        ref={canvasRef}
        className="rounded-lg shadow-md"
        style={{ width: '320px', height: '240px', display: showOverlay ? 'block' : 'none' }}
      />
      
      {/* Status indicators */}
      <div className="absolute top-2 right-2 flex gap-2">
        {/* Recording indicator */}
        <div className="bg-red-500 rounded-full w-3 h-3 animate-pulse" />
        
        {/* Face detection indicator */}
        {faceDetected ? (
          <div className="bg-green-500 rounded-full w-3 h-3" title="Face detected" />
        ) : (
          <div className="bg-yellow-500 rounded-full w-3 h-3" title="No face detected" />
        )}
      </div>

      {/* FPS counter */}
      {isActive && (
        <div className="absolute bottom-2 left-2 bg-black bg-opacity-50 text-white text-xs px-2 py-1 rounded">
          {fps} FPS
        </div>
      )}

      {/* Processing indicator */}
      {isProcessing && (
        <div className="absolute bottom-2 right-2 bg-blue-500 bg-opacity-75 text-white text-xs px-2 py-1 rounded">
          Processing...
        </div>
      )}
    </div>
  );
};

export default WebcamFeed;

