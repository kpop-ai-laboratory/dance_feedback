// src/components/Register.jsx
import React, { useState } from 'react';
import axios from 'axios';
import { useForm } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import * as yup from 'yup';
import { useNavigate, NavLink } from 'react-router-dom';

// 1. 스키마 정의
const schema = yup.object({
  username: yup.string(),
  email: yup.string().email('올바른 이메일을 입력하세요.').required('이메일을 입력해주세요.'),
  password: yup
    .string()
    .required('비밀번호를 입력해주세요.')
    .min(8, '비밀번호는 최소 8자 이상이어야 합니다.')
    .matches(/[0-9]/, '숫자를 하나 이상 포함해야 합니다.')
    .matches(/[A-Za-z]/, '영문자를 하나 이상 포함해야 합니다.'),
  confirm: yup
    .string()
    .oneOf([yup.ref('password')], '비밀번호가 일치하지 않습니다.')
    .required('비밀번호 확인을 입력해주세요.')
});

export default function Register() {
  // const [email, setEmail] = useState('');
  // const [username, setUsername] = useState('');
  // const [password, setPassword] = useState('');
  // const [confirm, setConfirm] = useState('');
  // const [error, setError] = useState(null);
  const navigate = useNavigate();

    // 2. useForm 훅 연결
    const {
      register,
      handleSubmit,
      formState: { errors, isSubmitting }
    } = useForm({
      resolver: yupResolver(schema)
    });

  // const handleSubmit = async e => {
  //   e.preventDefault();
  //   setError(null);
  //   if (password !== confirm) {
  //     setError('비밀번호가 일치하지 않습니다.');
  //     return;
  //   }

  //   try {
  //     const res = await axios.post(
  //       '/auth/register',
  //       { email, password, username },
  //       { withCredentials: true }
  //     );
  //     if (res.status === 201) {
  //       alert("회원가입이 완료되었습니다.")
  //       navigate('/login');
  //     }
  //   } catch (err) {
  //     if (err.response?.status === 409) {
  //       setError('이미 등록된 이메일입니다.');
  //       alert(error)
  //     } else {
  //       setError('회원가입 중 오류가 발생했습니다.');
  //     }
  //   }
  // };

   // 3. 제출 핸들러
   const onSubmit = async data => {
    try {
      const res = await axios.post('/auth/register', data, { withCredentials: true });
      if (res.status === 201) {
        alert('회원가입이 완료되었습니다.');
        navigate('/login');
      }
    } catch (err) {
      if (err.response?.status === 409) {
        alert('이미 등록된 이메일입니다.');
      } else {
        alert('회원가입 중 오류가 발생했습니다.');
      }
    }
  };

  return (
    <form
      //onSubmit={handleSubmit}
      onSubmit={handleSubmit(onSubmit)}
      className="bg-white/20 backdrop-blur-md border border-white/30 
       rounded-2xl p-10 w-full max-w-md mx-auto mt-16
       h-[730px]
       flex flex-col justify-between
      "
    >
      <h2 className="text-3xl font-bold text-white mb-6 text-center">
        Sign Up
      </h2>


      {/* Username */}

      <div className="mb-3">
      <label className="block text-white mb-2">Username</label>
      <input  {...register('username')}
        // type="text"
        // value={username}
        // onChange={e => setUsername(e.target.value)}

        className="w-full mb-4 px-4 py-3 rounded-lg bg-white/30 text-white placeholder-white/70 focus:outline-none"
        placeholder="Username"
        required
      />
      {errors.username && <p className="text-red-500 text-xs mt-1 min-h-[1rem]">{errors.username.message}</p>}
      </div>

      <div className="mb-3">
      <label className="block text-white mb-2">Email</label>
      <input {...register('email')} 
       // type="email"
       // value={email}
       // onChange={e => setEmail(e.target.value)}
        className="w-full mb-4 px-4 py-3 rounded-lg bg-white/30 text-white placeholder-white/70 focus:outline-none"
        placeholder="Email"
        required
      />
      {errors.email && <p className="text-red-500 text-xs mt-1 min-h-[1rem]">{errors.email.message}</p>}
      </div>

      <div className="mb-3">
      <label className="block text-white mb-2">Password</label>
      <input type="password" {...register('password')}
       // type="password"
       // value={password}
       // onChange={e => setPassword(e.target.value)}
        className="w-full mb-4 px-4 py-3 rounded-lg bg-white/30 text-white placeholder-white/70 focus:outline-none"
        placeholder="Password"
        required
      />
      {errors.password && <p className="text-red-500 text-xs mt-1 min-h-[1rem]">{errors.password.message}</p>}
      </div>

      <div className="mb-3">
      <label className="block text-white mb-2">Confirm Password</label>
      <input type="password" {...register('confirm')}
       // type="password"
       // value={confirm}
       // onChange={e => setConfirm(e.target.value)}
        className="w-full mb-6 px-4 py-3 rounded-lg bg-white/30 text-white placeholder-white/70 focus:outline-none"
        placeholder="Confirm Password"
        required
      />
      {errors.confirm && <p className="text-red-500 text-xs mt-1 min-h-[2rem]">{errors.confirm.message}</p>}
      </div>

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
