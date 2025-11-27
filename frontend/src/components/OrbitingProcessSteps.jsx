import React from 'react';

function OrbitingProcessSteps({ step, stepIndex, totalSteps, sequenceComplete, isDarkMode }) {
  if (!step) {
    return null;
  }

  return (
    <div className={`process-step-card ${isDarkMode ? 'is-dark' : 'is-light'}`} aria-live="polite">
      <div className="process-step-card__ring" aria-hidden="true" />
      <div className="process-step-card__content">
        <span className={`process-step-card__icon ${sequenceComplete ? 'is-complete' : ''}`} aria-hidden="true">
          {sequenceComplete ? '✓' : step.icon}
        </span>
        <div className="process-step-card__text">
          <p className="process-step-card__eyebrow">
            Step {Math.min(stepIndex + 1, totalSteps)} of {totalSteps}
          </p>
          <h4 className="process-step-card__title">{step.title}</h4>
          <p className="process-step-card__status">
            {sequenceComplete ? 'Completed · Generating output' : 'In progress'}
          </p>
        </div>
      </div>
    </div>
  );
}

export default OrbitingProcessSteps;

