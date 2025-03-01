import React, { useState } from 'react';

interface BiasAlertDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onNeutralize: () => void;
}

const BiasAlertDialog: React.FC<BiasAlertDialogProps> = ({ isOpen, onClose, onNeutralize }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 max-w-sm w-full">
        <h2 className="text-xl font-bold mb-4">Content Bias Alert</h2>
        <p className="mb-6">
          You've been scrolling through potentially biased content for a while. 
          Would you like to neutralize your feed?
        </p>
        <div className="flex justify-end space-x-4">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-200 text-gray-800 rounded hover:bg-gray-300"
          >
            Continue
          </button>
          <button
            onClick={() => {
              onNeutralize();
              onClose();
            }}
            className="px-4 py-2 bg-tiktok-cyan-dark text-white rounded hover:bg-tiktok-cyan"
          >
            Neutralize Feed
          </button>
        </div>
      </div>
    </div>
  );
};

export default BiasAlertDialog;