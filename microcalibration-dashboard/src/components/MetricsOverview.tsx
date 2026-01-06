'use client';

import { CalibrationDataPoint } from '@/types/calibration';
import { TrendingDown, Target, Clock, CheckCircle, Award, AlertTriangle } from 'lucide-react';

interface MetricsOverviewProps {
  data: CalibrationDataPoint[];
}

export default function MetricsOverview({ data }: MetricsOverviewProps) {
  if (data.length === 0) {
    return <div>No data available</div>;
  }

  // Find max epoch safely
  const maxEpoch = data.reduce((max, item) => Math.max(max, item.epoch), 0);
  
  // Get unique epochs and targets
  const allEpochs = Array.from(new Set(data.map(item => item.epoch))).sort((a, b) => a - b);
  const targetNames = Array.from(new Set(data.map(item => item.target_name)));
  
  // Get latest epoch data for final analysis
  const latestData = data.filter(item => item.epoch === maxEpoch);
  
  // Calculate convergence epoch
  const lossByEpoch = allEpochs.map(epoch => {
    const epochData = data.find(d => d.epoch === epoch);
    return { epoch, loss: epochData?.loss || 0 };
  });

  let convergenceEpoch: number | undefined;
  for (let i = 1; i < lossByEpoch.length; i++) {
    const currentLoss = lossByEpoch[i].loss;
    const prevLoss = lossByEpoch[i - 1].loss;
    const improvement = prevLoss > 0 ? (prevLoss - currentLoss) / prevLoss : 0;
    
    if (improvement < 0.001) {
      convergenceEpoch = lossByEpoch[i].epoch;
      break;
    }
  }

  // Calculate quality metrics
  const totalTargets = latestData.length;
  const avgRelAbsError = latestData.reduce((sum, item) => sum + item.rel_abs_error, 0) / totalTargets;
  const finalLoss = latestData[0]?.loss || 0;
  
  // Categorize targets by error quality
  const excellentCount = latestData.filter(item => item.rel_abs_error < 0.05).length;
  const goodCount = latestData.filter(item => item.rel_abs_error >= 0.05 && item.rel_abs_error < 0.20).length;
  const poorCount = latestData.filter(item => item.rel_abs_error >= 0.20).length;
  
  // Calculate overall quality score
  const qualityScore = ((excellentCount * 100 + goodCount * 75) / totalTargets).toFixed(1);
  
  // Determine quality status
  const getQualityStatus = () => {
    const score = parseFloat(qualityScore);
    if (score >= 90) return { label: 'Excellent', color: 'text-green-600', bg: 'bg-green-50' };
    if (score >= 75) return { label: 'Good', color: 'text-blue-600', bg: 'bg-blue-50' };
    if (score >= 60) return { label: 'Fair', color: 'text-yellow-600', bg: 'bg-yellow-50' };
    return { label: 'Needs work', color: 'text-red-600', bg: 'bg-red-50' };
  };

  const qualityStatus = getQualityStatus();

  return (
    <div className="space-y-6">
      {/* Main quality summary */}
      <div className="bg-white border border-gray-300 p-6 rounded-lg shadow-sm">
        <h2 className="text-2xl font-bold mb-2 text-gray-800">Calibration quality</h2>
        <p className="text-gray-600 mb-6">
          Assessment of how well calibrated weights match target statistics (final epoch: {maxEpoch})
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div className={`${qualityStatus.bg} p-4 rounded-lg border-l-4 ${qualityStatus.color.replace('text-', 'border-')}`}>
            <div className="flex items-center">
              <Award className={`h-8 w-8 ${qualityStatus.color} mr-3`} />
              <div>
                <div className="text-3xl font-bold text-gray-800">{qualityScore}%</div>
                <div className={`text-sm ${qualityStatus.color} font-medium`}>{qualityStatus.label} quality</div>
                <div className="text-xs text-gray-600 mt-1">
                  Weighted: Excellent (100%) + Good (75%)
                </div>
              </div>
            </div>
          </div>
          
          <div className="bg-gradient-to-r from-gray-50 to-gray-100 p-4 rounded-lg">
            <div className="flex items-center">
              <Target className="h-8 w-8 text-gray-600 mr-3" />
              <div>
                <div className="text-3xl font-bold text-gray-800">{avgRelAbsError.toFixed(4)}</div>
                <div className="text-sm text-gray-600">Avg relative error</div>
                <div className="text-xs text-gray-600 mt-1">
                  Target: &lt;0.05 • Current: {avgRelAbsError < 0.05 ? '✓ Good' : '⚠ Needs improvement'}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Error distribution */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="text-center p-3 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-green-600">{excellentCount}</div>
            <div className="text-sm text-gray-600 font-medium">Excellent</div>
            <div className="text-xs text-gray-500">&lt;5% error</div>
            <div className="text-xs text-green-600 font-medium">
              {((excellentCount / totalTargets) * 100).toFixed(0)}%
            </div>
          </div>
          <div className="text-center p-3 bg-yellow-50 rounded-lg">
            <div className="text-2xl font-bold text-yellow-600">{goodCount}</div>
            <div className="text-sm text-gray-600 font-medium">Good</div>
            <div className="text-xs text-gray-500">5-20% error</div>
            <div className="text-xs text-yellow-600 font-medium">
              {((goodCount / totalTargets) * 100).toFixed(0)}%
            </div>
          </div>
          <div className="text-center p-3 bg-red-50 rounded-lg">
            <div className="text-2xl font-bold text-red-600">{poorCount}</div>
            <div className="text-sm text-gray-600 font-medium">Needs work</div>
            <div className="text-xs text-gray-500">&gt;20% error</div>
            <div className="text-xs text-red-600 font-medium">
              {((poorCount / totalTargets) * 100).toFixed(0)}%
            </div>
          </div>
        </div>

        {/* Training statistics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm border-t pt-4">
          <div className="flex justify-between">
            <span className="text-gray-600">Total targets:</span>
            <span className="font-semibold text-gray-900">{targetNames.length}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Training epochs logged:</span>
            <span className="font-semibold text-gray-900">{allEpochs.length}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Final loss:</span>
            <span className="font-semibold text-gray-900">{finalLoss.toExponential(2)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Convergence:</span>
            <span className="font-semibold text-gray-900">
              {convergenceEpoch ? `Epoch ${convergenceEpoch}` : 'No early stop'}
            </span>
          </div>
        </div>
      </div>

      {/* Quick action cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-lg shadow border">
          <div className="flex items-center">
            <Clock className="h-6 w-6 text-blue-600" />
            <div className="ml-3">
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Duration</p>
              <p className="text-lg font-bold text-gray-900">{maxEpoch} epochs</p>
            </div>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg shadow border">
          <div className="flex items-center">
            <Target className="h-6 w-6 text-green-600" />
            <div className="ml-3">
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Targets</p>
              <p className="text-lg font-bold text-gray-900">{targetNames.length}</p>
            </div>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg shadow border">
          <div className="flex items-center">
            <TrendingDown className="h-6 w-6 text-red-600" />
            <div className="ml-3">
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Final loss</p>
              <p className="text-lg font-bold text-gray-900">{finalLoss.toExponential(1)}</p>
            </div>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg shadow border">
          <div className="flex items-center">
            {convergenceEpoch ? 
              <CheckCircle className="h-6 w-6 text-purple-600" /> : 
              <AlertTriangle className="h-6 w-6 text-orange-600" />
            }
            <div className="ml-3">
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Status</p>
              <p className="text-lg font-bold text-gray-900">
                {convergenceEpoch ? 'Converged' : 'In Progress'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
