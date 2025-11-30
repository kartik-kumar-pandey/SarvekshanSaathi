import React, { useState, useEffect } from 'react';
import ImageDisplay from './ImageDisplay';
import './ModelResults.css';

function ModelResults({ result, datasetName }) {
  const [classificationResult, setClassificationResult] = useState(null);
  const [loadingClassification, setLoadingClassification] = useState(false);
  const [classificationError, setClassificationError] = useState(null);

  // Clear classification results when dataset or result changes
  useEffect(() => {
    if (result && result.classification_image_url) {
      // Check if composite image exists in result.images
      const compositeImg = result.images?.find(img => img.name === 'Composite Visualization');
      const compositeUrl = compositeImg ? compositeImg.url : null;

      setClassificationResult({
        classification_image_url: result.classification_image_url,
        confusion_matrix_url: result.confusion_matrix_url,
        anomaly_score_map_url: result.anomaly_score_map_url,
        classification_report_url: result.classification_report_url,
        anomaly_stats: result.anomaly_stats || {},
        composite_url: compositeUrl
      });
      setClassificationError(null);
    } else {
      setClassificationResult(null);
      setClassificationError(null);
    }
  }, [datasetName, result]);

  if (!result) return null;

  // Filter out unwanted images
  const images = result.images ? result.images
    .filter(img => {
      const name = img.name || '';
      const unwanted = [
        'Anomaly Score Distribution',
        'Anomaly Intensity Histogram',
        't-SNE Visualization',
        'Confusion Matrix',
        'Anomaly Score Map',
        'PCA RGB Image',
        'Autoencoder Loss Curve',
        'Composite Visualization' // Filter out from main list as we show it separately
      ];
      // Check if name contains any of the unwanted strings (case insensitive)
      return !unwanted.some(u => name.toLowerCase().includes(u.toLowerCase()));
    })
    .map((img, index) => {
      let url = img.url || img.path || '';
      // Prepend backend base URL if url is relative and starts with /uploads/
      if (url.startsWith('/uploads/')) {
        url = `http://localhost:5000${url}`;
      }
      return {
        url,
        name: img.name || `Result ${index + 1}`,
        description: img.description
      };
    }) : [];

  const getFilename = (path) => {
    if (!path) return '';
    return path.split('/').pop();
  };

  const handleClassifyClick = async () => {
    if (!result.hsi_path || !result.gt_path || !datasetName) {
      setClassificationError('Missing required information for classification');
      return;
    }

    setLoadingClassification(true);
    setClassificationError(null);
    setClassificationResult(null);

    try {
      const response = await fetch('http://localhost:5000/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hsi_path: getFilename(result.hsi_path),
          gt_path: getFilename(result.gt_path),
          dataset_name: datasetName
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(errorData.error || `Server error: ${response.status}`);
      }

      const data = await response.json();

      // Find composite URL from data if available (though classify endpoint might need update to return it directly)
      // For now, construct it if not present, assuming naming convention
      const compositeUrl = data.composite_url || `/uploads/${datasetName}_composite.png`;

      // Store complete classification results
      setClassificationResult({
        classification_image_url: data.classification_image_url,
        confusion_matrix_url: data.confusion_matrix_url,
        // tsne_image_url: data.tsne_image_url, // disabled
        anomaly_score_map_url: data.anomaly_score_map_url,
        classification_report_url: data.classification_report_url,
        anomaly_stats: data.anomaly_stats || {},
        composite_url: compositeUrl
      });
    } catch (error) {
      setClassificationError(error.message || 'Classification failed');
      setClassificationResult(null);
    } finally {
      setLoadingClassification(false);
    }
  };

  const accuracy = result.stats && typeof result.stats.accuracy === 'number' ? result.stats.accuracy : null;

  return (
    <div className="model-results">
      <h2>Analysis Results</h2>

      {/* Display any numerical results or statistics */}
      {accuracy !== null && (
        <div className="results-stats">
          <h3>Statistics</h3>
          <div className="stats-section">
            <h4>Overall Accuracy</h4>
            <p>{(accuracy * 100).toFixed(2) + '%'}</p>
          </div>
        </div>
      )}

      {/* Display output images */}
      {images.length > 0 && (
        <ImageDisplay
          images={images}
          title="Analysis Output Images"
        />
      )}

      {/* Classify the anomalies button, shown only if detection results exist AND classification not already done */}
      {images.length > 0 && !classificationResult && (
        <div className="classification-section">
          <button onClick={handleClassifyClick} disabled={loadingClassification}>
            {loadingClassification ? 'Classifying...' : 'Classify the Anomalies'}
          </button>
          {classificationError && <p className="error-message">{classificationError}</p>}
        </div>
      )}

      {/* Display classification results */}
      {classificationResult && (
        <div className="classification-results">
          <h3>Classification Results - {datasetName.toUpperCase()}</h3>

          {classificationResult.anomaly_stats && (
            <div className="classification-stats">
              <p><strong>Anomalies Detected:</strong> {classificationResult.anomaly_stats.anomalies_detected || 'N/A'} ({classificationResult.anomaly_stats.anomaly_percentage?.toFixed(1) || 'N/A'}%)</p>
              <p><strong>Misclassifications:</strong> {classificationResult.anomaly_stats.misclassifications || 'N/A'}</p>
            </div>
          )}

          <div className="classification-images">
            {/* Composite Visualization (New) */}
            {classificationResult.composite_url && (
              <div className="classification-image-item full-width">
                <h4>Composite Visualization</h4>
                <img
                  src={classificationResult.composite_url.startsWith('http')
                    ? classificationResult.composite_url
                    : `http://localhost:5000${classificationResult.composite_url}`}
                  alt={`Composite Visualization - ${datasetName}`}
                  className="classification-image"
                  key={`composite-${datasetName}`}
                />
              </div>
            )}

            {classificationResult.classification_image_url && (
              <div className="classification-image-item">
                <h4>Predicted Classification Map</h4>
                <img
                  src={classificationResult.classification_image_url.startsWith('http')
                    ? classificationResult.classification_image_url
                    : `http://localhost:5000${classificationResult.classification_image_url}`}
                  alt={`Predicted Classification Map - ${datasetName}`}
                  className="classification-image"
                  key={`overlay-${datasetName}`}
                />
              </div>
            )}

            {/* Confusion Matrix hidden as per user request */}
            {/* 
                {classificationResult.confusion_matrix_url && (
                  <div className="classification-image-item">
                    <h4>Confusion Matrix</h4>
                    <img 
                      src={classificationResult.confusion_matrix_url.startsWith('http') 
                        ? classificationResult.confusion_matrix_url 
                        : `http://localhost:5000${classificationResult.confusion_matrix_url}`} 
                      alt={`Confusion Matrix - ${datasetName}`}
                      className="classification-image"
                      key={`confusion-${datasetName}`}
                    />
                  </div>
                )}
                */}

            {/* t-SNE Visualization disabled */}
            {false && classificationResult.tsne_image_url && (
              <div className="classification-image-item">
                <h4>t-SNE Visualization</h4>
                <img
                  src={classificationResult.tsne_image_url.startsWith('http')
                    ? classificationResult.tsne_image_url
                    : `http://localhost:5000${classificationResult.tsne_image_url}`}
                  alt={`t-SNE Visualization - ${datasetName}`}
                  className="classification-image"
                  key={`tsne-${datasetName}`}
                />
              </div>
            )}

            {/* Anomaly Score Map hidden as per user request */}
            {/*
                {classificationResult.anomaly_score_map_url && (
                  <div className="classification-image-item">
                    <h4>Anomaly Score Map</h4>
                    <img 
                      src={classificationResult.anomaly_score_map_url.startsWith('http') 
                        ? classificationResult.anomaly_score_map_url 
                        : `http://localhost:5000${classificationResult.anomaly_score_map_url}`} 
                      alt={`Anomaly Score Map - ${datasetName}`}
                      className="classification-image"
                      key={`anomaly-map-${datasetName}`}
                    />
                  </div>
                )}
                */}
          </div>
        </div>
      )}

      {/* Display any additional information */}
      {result.info && (
        <div className="results-info">
          <h3>Additional Information</h3>
          <p>{result.info}</p>
        </div>
      )}
    </div>
  );
}

export default ModelResults;
