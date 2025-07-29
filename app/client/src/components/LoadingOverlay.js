// src/components/LoadingOverlay.jsx
import React from 'react';

export default function LoadingOverlay({ isLoading, text }) {
  if (!isLoading) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex flex-col items-center justify-center z-50">
      <div className="flex space-x-2 mb-4">
        <span className="dot" />
        <span className="dot" />
        <span className="dot" />
      </div>
      <p className="text-white text-lg">{text}</p>
    </div>
  );
}
