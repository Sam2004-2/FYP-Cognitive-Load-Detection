import React, { useEffect, useRef, useState } from 'react';

interface WebcamFeedProps {
  isActive: boolean;
}

const WebcamFeed: React.FC<WebcamFeedProps> = ({ isActive }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    if (isActive) {
      navigator.mediaDevices
        .getUserMedia({ video: { width: 320, height: 240 } })
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
    }

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isActive]);

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
        style={{ width: '320px', height: '240px' }}
      />
      <div className="absolute top-2 right-2 bg-red-500 rounded-full w-3 h-3 animate-pulse" />
    </div>
  );
};

export default WebcamFeed;

