// src/AuthContext.jsx
import React, { createContext, useState , useEffect } from 'react';
import axios from 'axios';

export const AuthContext = createContext({
  currentUser: null,
  login: () => {},
  logout: () => {}
});

export function AuthProvider({ children }) {
  //const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);

   // 새로고침 포함, 앱 시작 시 한 번만 세션 확인
   useEffect(() => {
    axios
      .get('/auth/me', { withCredentials: true })
      .then(res => {
        if (res.data.authenticated) {
          // setIsAuthenticated(true);
          setCurrentUser(res.data.user);
          console.log(res.data.user)
        }
      })
      .catch(() => {
       // setIsAuthenticated(false);
       setCurrentUser(null);
      });
  }, []);

  const login = user => setCurrentUser(user);
  const logout = () => setCurrentUser(null);

  return (
    <AuthContext.Provider value={{ currentUser, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}
