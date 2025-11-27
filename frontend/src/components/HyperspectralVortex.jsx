import React from 'react';

const ringLayers = [
  { id: 'outer', duration: 32, direction: 'normal', blur: 24 },
  { id: 'mid', duration: 26, direction: 'reverse', blur: 18 },
  { id: 'inner', duration: 18, direction: 'normal', blur: 12 }
];

function HyperspectralVortex({ isDarkMode }) {
  return (
    <div className={`hyperspectral-vortex ${isDarkMode ? 'is-dark' : 'is-light'}`}>
      <div className="vortex-noise" aria-hidden="true" />
      <div className="vortex-core" aria-hidden="true">
        <div className="vortex-core__pulse" />
        <div className="vortex-core__flare" />
      </div>
      {ringLayers.map(layer => (
        <div
          key={layer.id}
          className={`vortex-ring vortex-ring--${layer.id}`}
          style={{
            animationDuration: `${layer.duration}s`,
            animationDirection: layer.direction,
            filter: `blur(${layer.blur}px)`
          }}
          aria-hidden="true"
        />
      ))}
      <div className="vortex-gradient" aria-hidden="true" />
    </div>
  );
}

export default HyperspectralVortex;

