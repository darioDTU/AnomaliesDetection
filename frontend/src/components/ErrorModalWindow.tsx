import type { Dispatch, SetStateAction } from 'react'

type ErrorModalWindowProps = {
  setHasPressButton: Dispatch<SetStateAction<boolean>>
  error: string
}

const ErrorModalWindow : React.FC<ErrorModalWindowProps> = ({ 
    setHasPressButton, 
    error
}) => (
  <div className="modal-overlay" onClick={() => setHasPressButton(false)}>
    <div className="modal-content error-modal" onClick={(e) => e.stopPropagation()}>
      <button className="modal-close" onClick={() => setHasPressButton(false)}>
        ×
      </button>
      <div className="modal-header error-header">
        <h2>Error</h2>
      </div>
      <div className="modal-body">
        <p>{error}</p>
      </div>
    </div>
  </div>
)

export default ErrorModalWindow
