import React, { useState, useEffect } from 'react';
import './TypingEffect.css';

const TypingEffect = ({
    text = [],
    speed = 100,
    eraseSpeed = 50,
    typingDelay = 1000,
    eraseDelay = 2000
}) => {
    const [displayedText, setDisplayedText] = useState('');
    const [currentTextIndex, setCurrentTextIndex] = useState(0);
    const [isDeleting, setIsDeleting] = useState(false);

    useEffect(() => {
        const handleTyping = () => {
            const currentFullText = text[currentTextIndex];

            if (isDeleting) {
                setDisplayedText(prev => prev.substring(0, prev.length - 1));
            } else {
                setDisplayedText(prev => currentFullText.substring(0, prev.length + 1));
            }

            if (!isDeleting && displayedText === currentFullText) {
                setTimeout(() => setIsDeleting(true), eraseDelay);
            } else if (isDeleting && displayedText === '') {
                setIsDeleting(false);
                setCurrentTextIndex((prev) => (prev + 1) % text.length);
            }
        };

        const timer = setTimeout(
            handleTyping,
            isDeleting ? eraseSpeed : speed
        );

        return () => clearTimeout(timer);
    }, [displayedText, isDeleting, currentTextIndex, text, speed, eraseSpeed, eraseDelay]);

    return (
        <span className="typing-effect">
            {displayedText}
            <span className="cursor">|</span>
        </span>
    );
};

export default TypingEffect;
