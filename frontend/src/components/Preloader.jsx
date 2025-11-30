'use client';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';
import React, { useRef, useState, useEffect } from 'react';
import './Preloader.css';
import logo from '../ss.jpg';

gsap.registerPlugin(useGSAP);

const Preloader = () => {
    const preloaderRef = useRef(null);
    const [isVisible, setIsVisible] = useState(true);

    useEffect(() => {
        // Add preloader-active class to body
        document.body.classList.add('preloader-active');

        // Set a timeout to hide preloader after 8 seconds (safety net, increased for logo)
        const hideTimeout = setTimeout(() => {
            document.body.classList.remove('preloader-active');
            window.scrollTo(0, 0);
            document.documentElement.scrollTop = 0;
            document.body.scrollTop = 0;
            setIsVisible(false);
        }, 8000);

        return () => {
            clearTimeout(hideTimeout);
            document.body.classList.remove('preloader-active');
        };
    }, []);

    useGSAP(
        () => {
            if (!preloaderRef.current) return;

            const tl = gsap.timeline();

            // 1. Animate the text
            tl.to('.name-text span', {
                y: 0,
                stagger: 0.1,
                duration: 0.3,
            });

            // 2. Hide text
            tl.to('.name-text span', {
                autoAlpha: 0,
                duration: 0.3,
                delay: 0.5
            });

            // 3. Animate Logo In (Zoom from center)
            tl.to('.preloader-logo', {
                autoAlpha: 1,
                scale: 1,
                duration: 1.2,
                ease: 'elastic.out(1, 0.5)',
                startAt: { scale: 0 }
            });

            // 4. Hold Logo
            tl.to('.preloader-logo', {
                duration: 1.0
            });

            // 5. Hide Logo (Zoom back to center/disappear)
            tl.to('.preloader-logo', {
                autoAlpha: 0,
                duration: 0.5,
                scale: 0,
                ease: 'back.in(1.7)'
            });

            // 6. Animate the preloader shutters
            tl.to('.preloader-item', {
                y: '100%',
                duration: 0.6,
                stagger: 0.1,
            }, '-=0.2');

            // 7. Cleanup
            tl.to(preloaderRef.current, {
                autoAlpha: 0,
                duration: 0.1,
                onComplete: () => {
                    document.body.classList.remove('preloader-active');
                    window.scrollTo(0, 0);
                    document.documentElement.scrollTop = 0;
                    document.body.scrollTop = 0;
                    setIsVisible(false);
                }
            });
        },
        { scope: preloaderRef },
    );

    if (!isVisible) {
        return null;
    }

    return (
        <div className="preloader-container" ref={preloaderRef}>
            <div className="preloader-item"></div>
            <div className="preloader-item"></div>
            <div className="preloader-item"></div>
            <div className="preloader-item"></div>
            <div className="preloader-item"></div>
            <div className="preloader-item"></div>
            <div className="preloader-item"></div>
            <div className="preloader-item"></div>
            <div className="preloader-item"></div>
            <div className="preloader-item"></div>

            <p className="name-text">
                <span>S</span>
                <span>a</span>
                <span>r</span>
                <span>v</span>
                <span>e</span>
                <span>k</span>
                <span>s</span>
                <span>h</span>
                <span>a</span>
                <span>n</span>
                <span>S</span>
                <span>a</span>
                <span>a</span>
                <span>t</span>
                <span>h</span>
                <span>i</span>
            </p>

            <img src={logo} alt="SarvekshanSaathi Logo" className="preloader-logo" />
        </div>
    );
};

export default Preloader;
