import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';
import IndexPage2 from './pages/IndexPage2';
import Upload2 from './pages/Upload2';
import TestPage from './pages/TestPage';
import SecTestPage from './pages/SecTestPage';
import TestViewPage from './pages/TestViewPage';
import TestFeedbackPage from './pages/TestFeedbackPage';
import CompareResult from './components/CompareResult';
import CompareResult1 from './components/CompareResult1';
import DemoResult from './components/DemoResult'
import DemoResult1 from './components/DemoResult1'
import Login from './components/Login';
import Register from './components/Register';



  function App() {
    return (
      <BrowserRouter>

        <div className="min-h-screen flex flex-col overflow-hidden bg-gradient-to-br from-orange-300 via-violet-400 to-rose-400">
       
        {/* <div className="min-h-screen relative overflow-hidden bg-gradient-to-br from-pink-500 via-purple-600 to-blue-600"> */}
        <Header />
        
        <main className="flex-grow">
          <Routes> 
          <Route path="/"       element={<IndexPage2 />} /> 
          <Route path="/upload"       element={<Upload2 />} /> 
          <Route path="/result" element={<CompareResult1 />} />        
          {/* 시연용  review */}
          <Route path="/frame-review"       element={<DemoResult1 />} /> 
          <Route path="/login"       element={<Login />} /> 
          <Route path="/register"       element={<Register />} /> 
          </Routes>
          </main>
        <Footer />
      </div>

      </BrowserRouter>
    );
  }
  
  export default App;
