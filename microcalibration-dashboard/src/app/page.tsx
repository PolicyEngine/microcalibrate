'use client';

import { useState } from 'react';
import FileUpload from '@/components/FileUpload';
import MetricsOverview from '@/components/MetricsOverview';
import LossChart from '@/components/LossChart';
import ErrorDistribution from '@/components/ErrorDistribution';
import CalibrationSummary from '@/components/CalibrationSummary';
import DataTable from '@/components/DataTable';
import { CalibrationDataPoint } from '@/types/calibration';
import { parseCalibrationCSV } from '@/utils/csvParser';

export default function Dashboard() {
  const [data, setData] = useState<CalibrationDataPoint[]>([]);
  const [filename, setFilename] = useState<string>('');
  const [error, setError] = useState<string>('');
  const [showDashboard, setShowDashboard] = useState<boolean>(false);

  const handleFileLoad = (content: string, name: string) => {
    try {
      const parsedData = parseCalibrationCSV(content);
      console.log('Parsed data length:', parsedData.length);
      setData(parsedData);
      setFilename(name);
      setError('');
      // Do not automatically show dashboard - let user click the button
    } catch (err) {
      console.error('Error parsing CSV:', err);
      setError(err instanceof Error ? err.message : 'Failed to parse CSV file');
      setData([]);
      setFilename('');
      setShowDashboard(false);
    }
  };

  console.log('Current state - showDashboard:', showDashboard, 'data.length:', data.length);

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2 text-gray-900">
            Calibration dashboard
          </h1>
          <p className="text-gray-600 text-lg">
            Microdata weight calibration assessment
          </p>
          {filename && (
            <p className="mt-1 text-sm text-blue-600">
              Loaded: {filename}
            </p>
          )}
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-md p-4">
            <div className="flex">
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">
                  Error loading file
                </h3>
                <div className="mt-2 text-sm text-red-700">
                  {error}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* File Upload */}
        {!showDashboard && (
          <div className="mb-8">
            <FileUpload 
              onFileLoad={handleFileLoad}
              onViewDashboard={() => {
                console.log('View Dashboard clicked, data length:', data.length);
                setShowDashboard(true);
              }}
            />
          </div>
        )}

        {/* Dashboard Content */}
        {showDashboard && (
          <div className="space-y-6">
            {/* Load New File Button */}
            <div className="flex justify-end">
              <button
                onClick={() => {
                  setData([]);
                  setFilename('');
                  setError('');
                  setShowDashboard(false);
                }}
                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md text-sm font-medium"
              >
                Load new file
              </button>
            </div>

            {data.length > 0 ? (
              <>
                {/* Metrics Overview */}
                <MetricsOverview data={data} />

                {/* Error Distribution */}
                <ErrorDistribution data={data} />

                {/* Calibration Summary */}
                <CalibrationSummary data={data} />

                {/* Loss Chart */}
                <LossChart data={data} />

                {/* Detailed Results Table */}
                <DataTable data={data} />
              </>
            ) : (
              <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
                <div className="text-yellow-800">
                  <h3 className="text-sm font-medium">No data loaded</h3>
                  <p className="text-sm mt-1">
                    Data length: {data.length}. There seems to be an issue with the data loading.
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}