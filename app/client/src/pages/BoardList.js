// src/pages/BoardList.jsx
import React, { useEffect, useState, useContext } from 'react'
import axios from 'axios'
import { Link, useNavigate } from 'react-router-dom'
import { AuthContext } from '../context/AuthContext'
import dayjs from 'dayjs'

export default function BoardList() {
  const { currentUser } = useContext(AuthContext)
  const navigate = useNavigate()

  const [posts, setPosts] = useState([])
  const [page, setPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const perPage = 10
  const maxPageButtons = 5

  useEffect(() => {
    axios
      .get('/board/posts', {
        params: { page, per_page: perPage },
        withCredentials: true
      })
      .then(res => {
        setPosts(res.data.posts)
       // console.log(res.data.posts)
        setTotalPages(
          res.data.total_pages
            ? res.data.total_pages
            : Math.ceil(res.data.total / perPage)
        )
      })
  }, [page])

  // 페이징 윈도우 계산 (최대 maxPageButtons개)
  const half = Math.floor(maxPageButtons / 2)
  let startPage = Math.max(1, page - half)
  let endPage = startPage + maxPageButtons - 1
  if (endPage > totalPages) {
    endPage = totalPages
    startPage = Math.max(1, endPage - maxPageButtons + 1)
  }
  const visiblePages = []
  for (let i = startPage; i <= endPage; i++) visiblePages.push(i)


  // 글쓰기 버튼
  const handleWriteClick = () => {
    if (!currentUser) {
      alert('로그인 후 글쓰기가 가능합니다.')
      navigate('/login')
    } else {
      navigate('/board/new')
    }
  }

  return (
    <div
      className="
        max-w-4xl mx-auto mt-12 p-8
        bg-white/20 backdrop-blur-md border border-white/30 rounded-2xl text-white
        flex flex-col
        h-[calc(100vh-250px)]
      "
    >
      {/* --- 헤더 --- */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">게시판</h1>
        <button
          onClick={handleWriteClick}
          className="px-4 py-2 bg-pink-500 hover:bg-pink-600 rounded-lg shadow"
        >
          글쓰기
        </button>
      </div>

      {/* --- 테이블 영역 (스크롤) --- */}
      <div
          className="
            flex-1 overflow-y-auto
            scrollbar-thin            /* 얇은 스크롤바 */
            scrollbar-thumb-purple-200 /* thumb 색상 */
            scrollbar-track-transparent /* 트랙은 투명 */
            scrollbar-thumb-rounded  /* 모서리 둥글게 */
          "
        >
        <table className="w-full table-auto text-left border-collapse">
          <thead>
            <tr className="border-b border-white/40">
              <th className="py-2 px-4">No</th>
              <th className="py-2 px-4">제목</th>
              <th className="py-2 px-4">작성일</th>
              <th className="py-2 px-2">조회수</th>
            </tr>
          </thead>
          <tbody>
            {posts.map(p => (
              <tr key={p.id} className="hover:bg-white/10">
                <td className="py-2 px-4">{p.id}</td>
                <td className="py-2 px-4">
                  <Link
                    to={`/board/${p.id}`}
                    className="underline hover:text-gray-200"
                  >
                    {p.title}
                  </Link>
                </td>
                <td className="py-2 px-4">
                  {dayjs(p.created_at).format('YYYY-MM-DD')}
                </td>
                <td className="py-2 px-4">{p.view_count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* --- 페이징 --- */}
      <nav className="mt-auto flex justify-center items-center space-x-1 pt-4">
        {/* << */}
        <button
          onClick={() => setPage(1)}
          disabled={page === 1}
          className="px-2 py-1 rounded hover:bg-white/30 disabled:opacity-50"
        >
          «
        </button>
        {/* < */}
        <button
          onClick={() => setPage(prev => Math.max(prev - 1, 1))}
          disabled={page === 1}
          className="px-2 py-1 rounded hover:bg-white/30 disabled:opacity-50"
        >
          ‹
        </button>

        {/* 숫자 페이지 */}
        {visiblePages.map(num => (
          <button
            key={num}
            onClick={() => setPage(num)}
            className={`
              px-3 py-1 rounded
              ${num === page
                ? 'bg-white text-purple-700 font-bold'
                : 'hover:bg-white/30'}
            `}
          >
            {num}
          </button>
        ))}

        {/* > */}
        <button
          onClick={() => setPage(prev => Math.min(prev + 1, totalPages))}
          disabled={page === totalPages}
          className="px-2 py-1 rounded hover:bg-white/30 disabled:opacity-50"
        >
          ›
        </button>
        {/* >> */}
        <button
          onClick={() => setPage(totalPages)}
          disabled={page === totalPages}
          className="px-2 py-1 rounded hover:bg-white/30 disabled:opacity-50"
        >
          »
        </button>
      </nav>
    </div>
  )
}
