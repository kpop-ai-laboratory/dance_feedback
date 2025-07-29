// src/components/Header.js
import React, { useState, useEffect } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import axios from 'axios';
import MAS from '../assets/MAS.png';

export default function Header() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const navigate = useNavigate();

  // 마운트 시 한 번만 인증 상태 확인
  useEffect(() => {
    axios
      .get('/auth/me', { withCredentials: true })
      .then(res => {
        setIsAuthenticated(res.data.authenticated);
      })
      .catch(() => {
        setIsAuthenticated(false);
      });
  }, []);

  // 로그아웃 핸들러
  const handleLogout = async () => {
    try {
      await axios.post('/auth/logout', {}, { withCredentials: true });
      setIsAuthenticated(false);
      navigate('/'); // 로그아웃 후 원하는 경로로 리다이렉트
    } catch (err) {
      console.error('Logout error', err);
    }
  };

  const routes = [
    { label: 'Home',    path: '/' },
    { label: 'Upload',  path: '/upload' },
    { label: 'FrameReview',  path: '/frame-review' },
    { label: 'SequenceReview', path: '/sequence-review' },
  ];

  return (
    <header className="relative z-10 p-6 flex justify-between items-center text-white drop-shadow-lg">
      {/* Background circles */}
      <div className="absolute inset-0 pointer-events-none -z-10">
        <div className="absolute top-10 left-10 w-20 h-20 bg-yellow-200 rounded-full opacity-30 animate-pulse" />
        <div className="absolute bottom-12 right-1/3 w-24 h-24 bg-pink-200 rounded-full opacity-25 animate-bounce delay-500" />
      </div>

      {/* 로고 + 타이틀 */}
      <div className="flex items-center space-x-2">
        <img src={MAS} alt="Move AI Scan" className="h-8 w-auto" />
        <span className="text-lg font-bold">Move Ai Scan</span>
      </div>

      {/* 네비게이션 */}
      <nav className="hidden md:flex items-center space-x-6 text-white/200">
        {routes.map(({ label, path }) => (
          <NavLink key={label} to={path}>
            {label}
          </NavLink>
        ))}

        {/* 로그인 상태에 따라 Login / Logout */}
        {!isAuthenticated ? (
          <NavLink to="/login">Login</NavLink>
        ) : (
          <button
            onClick={handleLogout}
            className="text-white hover:underline"
          >
            Logout
          </button>
        )}
      </nav>
    </header>
  );
}
