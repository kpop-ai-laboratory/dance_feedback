// src/components/Register.jsx
import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate, NavLink } from 'react-router-dom';


export default function Register() {
  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleSubmit = async e => {
    e.preventDefault();
    setError(null);
    if (password !== confirm) {
      setError('비밀번호가 일치하지 않습니다.');
      return;
    }

    try {
      const res = await axios.post(
        '/auth/register',
        { email, password, username },
        { withCredentials: true }
      );
      if (res.status === 201) {
        alert("회원가입이 완료되었습니다.")
        navigate('/login');
      }
    } catch (err) {
      if (err.response?.status === 409) {
        setError('이미 등록된 이메일입니다.');
        alert(error)
      } else {
        setError('회원가입 중 오류가 발생했습니다.');
      }
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="bg-white/20 backdrop-blur-md border border-white/30 rounded-2xl p-10 w-full max-w-md mx-auto mt-32"
    >
      <h2 className="text-3xl font-bold text-white mb-6 text-center">
        Sign Up
      </h2>



      <label className="block text-white mb-2">Username</label>
      <input
        type="text"
        value={username}
        onChange={e => setUsername(e.target.value)}
        className="w-full mb-4 px-4 py-3 rounded-lg bg-white/30 text-white placeholder-white/70 focus:outline-none"
        placeholder="Username"
        required
      />

      <label className="block text-white mb-2">Email</label>
      <input
        type="email"
        value={email}
        onChange={e => setEmail(e.target.value)}
        className="w-full mb-4 px-4 py-3 rounded-lg bg-white/30 text-white placeholder-white/70 focus:outline-none"
        placeholder="Email"
        required
      />



      <label className="block text-white mb-2">Password</label>
      <input
        type="password"
        value={password}
        onChange={e => setPassword(e.target.value)}
        className="w-full mb-4 px-4 py-3 rounded-lg bg-white/30 text-white placeholder-white/70 focus:outline-none"
        placeholder="Password"
        required
      />

      <label className="block text-white mb-2">Confirm Password</label>
      <input
        type="password"
        value={confirm}
        onChange={e => setConfirm(e.target.value)}
        className="w-full mb-6 px-4 py-3 rounded-lg bg-white/30 text-white placeholder-white/70 focus:outline-none"
        placeholder="Confirm Password"
        required
      />

      <button
        type="submit"
        className="w-full py-4 bg-gradient-to-r from-pink-400 to-rose-500 hover:from-pink-500 hover:to-rose-600 text-white font-bold rounded-xl shadow-lg transition"
      >
        REGISTER
      </button>

      <p className="mt-4 text-center text-white/80">
      Already have account?{' '}
        <NavLink to="/login" className="font-bold underline">
        Sign In
        </NavLink>
      </p>
    </form>
  );
}
