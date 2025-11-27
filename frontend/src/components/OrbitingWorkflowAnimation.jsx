import React, { useMemo } from 'react';
import PipelineProgress from './PipelineProgress';

const particleCount = 24;

const OrbitingWorkflowAnimation = ({
  steps,
  activeStepIndex = 0,
  sequenceComplete = false,
  isDarkMode = false
}) => {
const particles = useMemo(
  () =>
    Array.from({ length: particleCount }).map((_, index) => ({
      id: `particle-${index}`,
      delay: `${index * 0.35}s`,
      x: `${(Math.random() - 0.5) * 60}%`,
      y: `${(Math.random() - 0.5) * 60}%`
    })),
  []
);

if (!steps?.length) {
  return null;
}

const clampedIndex = Math.min(activeStepIndex, steps.length - 1);
const highlightIndex = sequenceComplete ? steps.length - 1 : clampedIndex;
const activeStep = steps[clampedIndex];

  return (
    <div
      className={`orbiting-workflow ${isDarkMode ? 'is-dark' : 'is-light'}`}
      role="status"
      aria-live="assertive"
    >
      <div className="orbiting-workflow__background" aria-hidden="true">
        <div className="orbiting-workflow__gridlines" />
        <div className="orbiting-workflow__particles">
          {particles.map((particle) => (
            <span
              key={particle.id}
              className="orbiting-workflow__particle"
              style={{
                '--delay': particle.delay,
                '--x': particle.x,
                '--y': particle.y
              }}
            />
          ))}
        </div>
      </div>

      <div className="orbiting-workflow__core">
        <div className="orbit-core__halo" />
        <div className="orbit-core__pulse" />
        <div className="orbit-core__ring">
          <div className="orbit-core__glow" />
        </div>
        
      </div>

      

      <div className="orbiting-workflow__status">
        <p className="loading-eyebrow">Orbiting workflow</p>
        <h3 className="loading-title">
          {sequenceComplete ? 'Finalizing anomaly maps' : activeStep?.title}
        </h3>
        <p className="loading-subtext">
          {sequenceComplete
            ? 'All spectral nodes synced · awaiting backend completion'
            : `Node ${clampedIndex + 1} of ${steps.length} · signals flowing in real time`}
        </p>
        <PipelineProgress
          isDarkMode={isDarkMode}
          activeStepIndex={clampedIndex}
          steps={steps}
          sequenceComplete={sequenceComplete}
        />
      </div>
    </div>
  );
};

export default OrbitingWorkflowAnimation;


