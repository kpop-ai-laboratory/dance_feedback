// src/pages/BoardForm.jsx
import React, { useEffect, useState, useRef } from 'react';

import { useParams, useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import { Editor } from '@toast-ui/react-editor';
import '@toast-ui/editor/dist/toastui-editor.css';
import MediaUpload from '../components/MediaUpload';

export default function BoardForm() {
  const { id } = useParams();
  const isEdit = Boolean(id);
  const [title, setTitle] = useState('');
  const navigate = useNavigate();
  const editorRef = useRef();


  useEffect(() => {
    if (!isEdit) return;
    axios.get(`/board/posts/${id}`, { withCredentials: true })
      .then(res => {
        setTitle(res.data.title);
        editorRef.current.getInstance().setHTML(res.data.content);
      })
      .catch(() => navigate('/board'));
  }, [id, isEdit, navigate]);

  const handleUpload = urls => {
    const inst = editorRef.current.getInstance();

    urls.forEach(u => {
      // 이미지인지 동영상인지 판정
      const tag = /\.(mp4|mov|webm)$/i.test(u)
        ? `<video src="${u}" controls class="rounded-lg w-full mb-4"></video>`
        : `<img src="${u}" class="rounded-lg w-full mb-4" />`;

      // 1) 현재 HTML 가져오기
      const currentHtml = inst.getHTML();
      // 2) 새 태그를 뒤에 붙이기
      const newHtml = currentHtml + tag;
      // 3) 에디터에 다시 설정
      inst.setHTML(newHtml);
    });
  };

  const handleSubmit = async e => {
    e.preventDefault();
    const content = editorRef.current.getInstance().getHTML();
    const payload = { title, content };

    if (isEdit) {
      await axios.put(`/board/posts/${id}`, payload, { withCredentials: true });
    } else {
      await axios.post('/board/posts', payload, { withCredentials: true });
    }
    navigate('/board');
  };

  // useEffect(() => {
  //   if (!isEdit) return
  //   axios.get(`/board/posts/${id}`, { withCredentials: true })
  //     .then(res => {
  //       setTitle(res.data.title)
  //       editorRef.current.getInstance().setHTML(res.data.content)
  //       // DB에 저장된 <video> 태그는 무시하고, 미리보기용 URL만 다시 세팅할 수도 있습니다.
  //     })
  //     .catch(() => navigate('/board'))
  // }, [id, isEdit, navigate])

  // // MediaUpload에서 업로드된 URL만 꺼내서 state에 저장
  // const handleUpload = urls => {
  //   setVideoUrls(urls)
  // }

  // const handleSubmit = async e => {
  //   e.preventDefault()
  //   const content = editorRef.current.getInstance().getHTML()
  //   const payload = { title, content }
  //   if (isEdit) {
  //     await axios.put(`/board/posts/${id}`, payload, { withCredentials: true })
  //   } else {
  //     await axios.post('/board/posts', payload, { withCredentials: true })
  //   }
  //   navigate('/board')
  // }

  return (
    <div className="max-w-3xl mx-auto mt-12">
      <form
        onSubmit={handleSubmit}
        className="bg-white/20 backdrop-blur-md border border-white/30 rounded-2xl p-8 space-y-6 text-white"
      >

      {/* 절대 위치 버튼 */}


        <h1 className="text-2xl font-bold">
          {isEdit ? '게시글 수정' : '새 글 작성'}

          <Link
        to="/board"
        className="absolute top-4 right-4 text-white/90 hover:text-white"
      >
        ← 
      </Link>
        </h1>

        <input
          type="text"
          value={title}
          onChange={e => setTitle(e.target.value)}
          placeholder="제목을 입력하세요"
          className="w-full p-3 bg-white/10 rounded-lg placeholder-white/70 focus:outline-none"
          required
        />

        <MediaUpload onUpload={handleUpload} />

        <div className="bg-white/10 rounded-lg overflow-hidden">
          <Editor
            ref={editorRef}
            initialValue=""
            previewStyle="vertical"
            height="300px"
            initialEditType="wysiwyg"
            useCommandShortcut={true}
          />
        </div>


        <button
          type="submit"
          className="w-full py-3 bg-gradient-to-r from-pink-400 to-rose-500 hover:from-pink-500 hover:to-rose-600 rounded-xl font-bold shadow-lg transition"
        >
          {isEdit ? '수정 완료' : '등록하기'}
        </button>
      </form>
    </div>
  );
}
