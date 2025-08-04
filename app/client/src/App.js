import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';
import IndexPage from './pages/IndexPage';
import Upload from './pages/Upload';
import About from './pages/About';

import CompareResult from './components/CompareResult';
import SequenceReviewPage from './pages/SequenceReviewPage'
import DemoSeq from './components/DemoSeq';


import DemoResult from './components/DemoResult'
import Login from './components/Login';
import Register from './components/Register';
import BoardList   from './pages/BoardList';
import BoardDetail from './pages/BoardDetail';
import BoardForm   from './pages/BoardForm';



  function App() {
    return (
      <BrowserRouter>

        <div className="min-h-screen flex flex-col overflow-hidden bg-gradient-to-br from-orange-300 via-violet-400 to-rose-400">
       
        {/* <div className="min-h-screen relative overflow-hidden bg-gradient-to-br from-pink-500 via-purple-600 to-blue-600"> */}
        <Header />
        
        <main className="flex-grow">
          <Routes> 
          <Route path="/"       element={<IndexPage />} /> 
          <Route path="/about"       element={<About />} /> 
          <Route path="/upload"       element={<Upload />} /> 
          
          
          <Route path="/result" element={<CompareResult />} />      
          <Route path="/sequence-result"  element={<SequenceReviewPage />} />    
          
          
          
          {/* 시연용  review */}
          <Route path="/frame-review"       element={<DemoResult />} /> 
          <Route path="/sequence-review" element={<DemoSeq />} /> 


          <Route path="/login"       element={<Login />} /> 
          <Route path="/register"       element={<Register />} /> 
          
          <Route path="/board"            element={<BoardList />} />
          <Route path="/board/new"        element={<BoardForm />} />
          <Route path="/board/:id"        element={<BoardDetail />} />
          <Route path="/board/:id/edit"   element={<BoardForm />} />
          </Routes>
          </main>
        <Footer />
      </div>

      </BrowserRouter>
    );
  }
  
  export default App;
