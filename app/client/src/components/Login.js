// src/components/Login.jsx
import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate, NavLink } from 'react-router-dom';

export default function Login() {
  const [userEmail, setUserEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleSubmit = async e => {
    e.preventDefault();
    try {
      const res = await axios.post(
        '/auth/login',
        { userEmail, password },
        { withCredentials: true }
      );
      if (res.data.success) {
        console.log(res.data)
        navigate('/');
      }
    } catch (err) {
      setError('아이디 또는 비밀번호가 올바르지 않습니다.');
      alert(error)
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="bg-white/20 backdrop-blur-md border border-white/30 rounded-2xl p-10 w-full max-w-md mx-auto mt-32"
    >
      <h2 className="text-3xl font-bold text-white mb-6 text-center">
        Move Ai Scan
      </h2>

      {/* {error && <p className="text-red-300 mb-4 text-center">{error}</p>} */}

      <label className="block text-white mb-2">Email</label>
      <input
        type="text"
        value={userEmail}
        onChange={e => setUserEmail(e.target.value)}
        className="w-full mb-4 px-4 py-3 rounded-lg bg-white/30 text-white placeholder-white/70 focus:outline-none"
        placeholder="Email"
      />

      <label className="block text-white mb-2">Password</label>
      <input
        type="password"
        value={password}
        onChange={e => setPassword(e.target.value)}
        className="w-full mb-6 px-4 py-3 rounded-lg bg-white/30 text-white placeholder-white/70 focus:outline-none"
        placeholder="Password"
      />

      <button
        type="submit"
        className="w-full py-4 bg-gradient-to-r from-pink-400 to-rose-500 hover:from-pink-500 hover:to-rose-600 text-white font-bold rounded-xl shadow-lg transition"
      >
        SIGN IN
      </button>

      <p className="mt-4 text-center text-white/80">
        Don't you have account?{' '}
        <NavLink to="/register" className="font-bold underline">
        Sign Up Here
        </NavLink>
      </p>
    </form>
  );
}