// src/pages/BoardDetail.jsx
import React, { useEffect, useState, useContext } from 'react';
import axios from 'axios';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { AuthContext } from '../context/AuthContext'
import dayjs from 'dayjs'
import utc    from 'dayjs/plugin/utc';


dayjs.extend(utc);

export default function BoardDetail() {
  const { id } = useParams();
  const [post, setPost] = useState(null);
  const { currentUser } = useContext(AuthContext);
  const nav = useNavigate();
  


  useEffect(() => {
    axios.get(`/board/posts/${id}`, { withCredentials: true })
      .then(
        res => { 
        // console.log(res)  
        console.log(currentUser)
        setPost(res.data)
        console.log(res.data)
        }      
      )
      .catch(() => nav('/board'));
  }, [id, nav]);

  

  if (!post) return <div className="text-center p-10 text-white">Loading…</div>;

  const isMine = currentUser?.id === post.user_id;

  return (
    <div className="max-w-3xl mx-auto mt-12 p-8 bg-white/20 backdrop-blur-md border border-white/30 rounded-2xl text-white">
      <h1 className="text-4xl font-extrabold mb-4 drop-shadow-lg">{post.title}</h1>
      <hr />
      <div className="text-sm  text-white/70 mb-6 mt-3">
        {/* dayjs.utc() 로, 서버에서 받은 GMT 시간을 그대로 해석 */}
        {dayjs.utc(post.updated_at).format('YYYY-MM-DD HH:mm:ss')}
        </div>
      <article
        className="prose prose-lg prose-invert mb-8"
        dangerouslySetInnerHTML={{ __html: post.content }}
      />
      <div className="flex items-center w-full">
      <div className="flex-1 flex justify-center space-x-6">
        
        
        {isMine && (
          <>
            <Link
              to={`/board/${id}/edit`}
              className="
                px-4 py-2
                bg-gradient-to-r from-blue-400 to-blue-600
                hover:from-blue-500 hover:to-blue-500
                text-white font-semibold
                rounded-xl shadow-lg
                transition
              "
            >
              수정
          </Link>

          <button
            onClick={async () => {
              if (window.confirm('정말 삭제하시겠습니까?')) {
                await axios.delete(`/board/posts/${id}`, { withCredentials: true });
                nav('/board');
              }
            }}
            className="
                px-4 py-2
                bg-gradient-to-r from-pink-400 to-rose-500
                hover:from-pink-500 hover:to-rose-600
                text-white font-semibold
                rounded-xl shadow-lg
                transition
          "
          >
            삭제
          </button>
          </>
    )}


      </div>
      <Link
          to="/board"
          className="px-4 py-2 bg-white/20 hover:bg-white/30 rounded-xl shadow text-white"
        >
          목록
        </Link>
      </div>

    </div>
  );
}
