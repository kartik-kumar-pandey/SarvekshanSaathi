import React, { useMemo, useEffect, useRef } from 'react';

const DEFAULT_STEPS = [
  'Training Reconstruction Mechanism',
  'Extracting Spectral Patches',
  'Running Classification Mechanism',
  'Detecting Anomalies',
  'Refining Output',
  'Generating Heatmaps',
  'Finalizing Results'
];

function PipelineProgress({
  steps = DEFAULT_STEPS,
  activeStepIndex = 0,
  sequenceComplete = false,
  isDarkMode = false
}) {
  const normalizedSteps = useMemo(() => {
    const source = steps && steps.length ? steps : DEFAULT_STEPS;
    return source.map((step) => (typeof step === 'string' ? step : step?.title || ''));
  }, [steps]);

  const currentIndex = sequenceComplete
    ? normalizedSteps.length - 1
    : Math.min(Math.max(activeStepIndex, 0), normalizedSteps.length - 1);

  const viewportRef = useRef(null);
  const stepRefs = useRef([]);

  useEffect(() => {
    const viewport = viewportRef.current;
    const targetNode = stepRefs.current[currentIndex];

    if (!viewport || !targetNode) {
      return;
    }

    const targetCenter = targetNode.offsetLeft + targetNode.offsetWidth / 2;
    const viewportWidth = viewport.offsetWidth;
    const scrollLeft = Math.max(targetCenter - viewportWidth / 2, 0);

    viewport.scrollTo({
      left: scrollLeft,
      behavior: 'smooth'
    });
  }, [currentIndex]);

  const stepStates = useMemo(() => (
    normalizedSteps.map((label, index) => {
      if (sequenceComplete || index < currentIndex) {
        return { label, state: 'complete' };
      }
      if (index === currentIndex) {
        return { label, state: 'active' };
      }
      return { label, state: 'inactive' };
    })
  ), [normalizedSteps, currentIndex, sequenceComplete]);

  return (
    <div
      className={`pipeline-progress ${isDarkMode ? 'is-dark' : 'is-light'}`}
      role="presentation"
      aria-label="Pipeline progress indicator"
    >
      <div className="pipeline-progress__viewport" ref={viewportRef}>
        <div className="pipeline-progress__track">
          {stepStates.map(({ label, state }, index) => (
            <div
              key={`pipeline-node-${label || index}`}
              className={`pipeline-progress__step ${state === 'active'
                  ? 'pipeline-progress__step--active'
                  : state === 'complete'
                    ? 'pipeline-progress__step--complete'
                    : ''
                }`}
            >
              <div
                className={`pipeline-progress__node ${state === 'active'
                    ? 'pipeline-progress__node--active'
                    : state === 'complete'
                      ? 'pipeline-progress__node--complete'
                      : ''
                  }`}
                aria-current={state === 'active' ? 'step' : undefined}
                ref={(element) => {
                  stepRefs.current[index] = element;
                }}
              >
                <span className="pipeline-progress__halo" aria-hidden="true" />
                <span className="pipeline-progress__dot" aria-hidden="true">
                  {index + 1}
                </span>
              </div>
              <span className="pipeline-progress__label">{label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default PipelineProgress;


