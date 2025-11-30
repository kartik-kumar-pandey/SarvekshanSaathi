import React, { useState } from 'react';

function FileUpload({ selectedDataset, onUploadSuccess, onUploadFailure, isLoading, setIsLoading, onProgress }) {
  const [hsiFile, setHsiFile] = useState(null);
  const [gtFile, setGtFile] = useState(null);
  const [internalLoading, setInternalLoading] = useState(false);

  const loading = typeof isLoading === 'boolean' ? isLoading : internalLoading;

  const startLoading = () => {
    setInternalLoading(true);
    if (setIsLoading) {
      setIsLoading(true);
    }
  };

  const stopLoading = () => {
    setInternalLoading(false);
    if (setIsLoading) {
      setIsLoading(false);
    }
  };

  const handleHsiChange = (e) => {
    const file = e.target.files[0];
    console.log('HSI file selected:', file?.name, 'type:', file?.type, 'size:', file?.size);
    setHsiFile(file);
  };

  const handleGtChange = (e) => {
    const file = e.target.files[0];
    console.log('GT file selected:', file?.name, 'type:', file?.type, 'size:', file?.size);
    setGtFile(file);
  };

  const handleSubmit = async () => {
    try {
      if (!hsiFile || !gtFile) {
        onUploadFailure('Please select both HSI and Ground Truth files!');
        return;
      }
      if (!selectedDataset) {
        onUploadFailure('Please select a dataset first!');
        return;
      }

      // Check file extensions
      if (!hsiFile.name.endsWith('.mat') || !gtFile.name.endsWith('.mat')) {
        onUploadFailure('Both files must be .mat files!');
        return;
      }

      // Check file sizes (16MB limit)
      const maxSize = 60 * 1024 * 1024; // 16MB in bytes
      if (hsiFile.size > maxSize || gtFile.size > maxSize) {
        onUploadFailure('Files must be smaller than 16MB!');
        return;
      }

      startLoading();
      console.log('Starting upload with dataset:', selectedDataset);
      console.log('HSI file:', hsiFile.name, 'size:', hsiFile.size);
      console.log('GT file:', gtFile.name, 'size:', gtFile.size);

      const formData = new FormData();
      formData.append('hsi_file', hsiFile);
      formData.append('gt_file', gtFile);
      formData.append('dataset_name', selectedDataset);

      const response = await fetch('http://127.0.0.1:5000/upload', {
        method: 'POST',
        body: formData,
      });

      console.log('Response status:', response.status);
      const data = await response.json();
      console.log('Server response:', data);

      if (response.ok) {
        const taskId = data.task_id;
        console.log('Upload successful, task ID:', taskId);

        // Start polling for status
        const pollInterval = setInterval(async () => {
          try {
            const statusResponse = await fetch(`http://127.0.0.1:5000/status/${taskId}`);
            const statusData = await statusResponse.json();

            if (statusResponse.ok) {
              console.log('Task status:', statusData.status, 'Step:', statusData.step);

              if (statusData.status === 'processing') {
                if (onProgress && typeof onProgress === 'function') {
                  onProgress(statusData.step);
                }
              } else if (statusData.status === 'completed') {
                clearInterval(pollInterval);
                onUploadSuccess(statusData.results);
                stopLoading();
              } else if (statusData.status === 'failed') {
                clearInterval(pollInterval);
                onUploadFailure(statusData.error || 'Processing failed');
                stopLoading();
              }
            } else {
              console.error('Error checking status:', statusData.error);
            }
          } catch (err) {
            console.error('Polling error:', err);
          }
        }, 1000); // Poll every 1 second

      } else {
        onUploadFailure(data.error || 'Upload failed');
        stopLoading();
      }
    } catch (error) {
      console.error('Upload error:', error);
      if (error.name === 'TypeError' && error.message === 'Failed to fetch') {
        onUploadFailure('Could not connect to the server. Please make sure the backend is running.');
      } else {
        onUploadFailure(error.message || 'Upload failed due to network error');
      }
      stopLoading();
    }
  };

  return (
    <div className="file-upload-container">
      <div className="file-group">
        <label htmlFor="hsi-upload">HSI File (.mat):</label>
        <input
          id="hsi-upload"
          type="file"
          onChange={handleHsiChange}
          disabled={loading}
          className="file-input"
          accept=".mat"
        />
        {hsiFile && (
          <div className="file-info">
            Size: {(hsiFile.size / (1024 * 1024)).toFixed(2)}MB
          </div>
        )}
      </div>

      <div className="file-group">
        <label htmlFor="gt-upload">Ground Truth File (.mat):</label>
        <input
          id="gt-upload"
          type="file"
          onChange={handleGtChange}
          disabled={loading}
          className="file-input"
          accept=".mat"
        />
        {gtFile && (
          <div className="file-info">
            Size: {(gtFile.size / (1024 * 1024)).toFixed(2)}MB
          </div>
        )}
      </div>

      <button
        onClick={handleSubmit}
        className="upload-btn"
        disabled={loading}
      >
        {loading ? (
          <>
            Uploading...
            <svg
              className="svg-spinner"
              width="16px"
              height="16px"
              viewBox="0 0 66 66"
              xmlns="http://www.w3.org/2000/svg"
            >
              <circle
                className="path"
                fill="none"
                strokeWidth="6"
                strokeLinecap="round"
                cx="33"
                cy="33"
                r="30"
              />
            </svg>
          </>
        ) : (
          'Upload'
        )}
      </button>
    </div>
  );
}

export default FileUpload;
