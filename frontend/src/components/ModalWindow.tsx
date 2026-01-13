import { ENDPOINTS } from '../service/endpoints.ts'
import { BASE_URL } from '../service/api.ts'

import type { Dispatch, SetStateAction } from 'react'

export type Statistics = Record<string, string>

type ModalWindowProps = {
  setHasPressButton: Dispatch<SetStateAction<boolean>>
  stats: Statistics
}

const ModalWindow = ({ setHasPressButton, stats }: ModalWindowProps) => (
  <div className="modal-overlay" onClick={() => setHasPressButton(false)}>
    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
      <button className="modal-close" onClick={() => setHasPressButton(false)}>
        ×
      </button>
      <div className="modal-header">
        <h2>Analysis Results</h2>
      </div>
      <div className="modal-body">
        <img
          src={`${BASE_URL}/${ENDPOINTS.showImage}?t=${Date.now()}`}
          alt="API result"
          className="modal-image"
        />
        <div className="modal-stats">
          <h4>Statistics</h4>
          <ul>
            {stats ? 
            Object.entries(stats).map(([key, value]) => (
            <li key={key}>
                <strong>{key}:</strong> {value as string}
            </li>
            ))
            : <li>No statistics available for area with no anomalies</li>}
          </ul>
        </div>
      </div>
    </div>
  </div>
)

export default ModalWindow