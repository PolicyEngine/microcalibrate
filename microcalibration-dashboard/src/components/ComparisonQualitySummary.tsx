'use client';

import { CalibrationDataPoint } from '@/types/calibration';
import { TrendingUp, TrendingDown, Minus, Scale } from 'lucide-react';

interface ComparisonQualitySummaryProps {
  firstData: CalibrationDataPoint[];
  secondData: CalibrationDataPoint[];
  firstName: string;
  secondName: string;
}

interface TargetComparison {
  target_name: string;
  firstFinalError: number;
  secondFinalError: number;
  improvement: number;
  category: 'improved_significantly' | 'worsened' | 'minimal_change';
}

export default function ComparisonQualitySummary({ 
  firstData, 
  secondData, 
  firstName, 
  secondName 
}: ComparisonQualitySummaryProps) {
  if (firstData.length === 0 || secondData.length === 0) {
    return null;
  }

  // Get final epoch data for both datasets
  const getMaxEpoch = (data: CalibrationDataPoint[]) => {
    let maxEpoch = -Infinity;
    for (const point of data) {
      if (point.epoch > maxEpoch) {
        maxEpoch = point.epoch;
      }
    }
    return maxEpoch;
  };

  const firstMaxEpoch = getMaxEpoch(firstData);
  const secondMaxEpoch = getMaxEpoch(secondData);
  
  const firstFinalData = firstData.filter(d => d.epoch === firstMaxEpoch);
  const secondFinalData = secondData.filter(d => d.epoch === secondMaxEpoch);

  // Compare overlapping targets
  const targetComparisons: TargetComparison[] = [];
  
  firstFinalData.forEach(firstTarget => {
    const secondTarget = secondFinalData.find(s => s.target_name === firstTarget.target_name);
    
    if (secondTarget && 
        firstTarget.rel_abs_error !== undefined && 
        secondTarget.rel_abs_error !== undefined &&
        !isNaN(firstTarget.rel_abs_error) && 
        !isNaN(secondTarget.rel_abs_error)) {
      
      const firstError = firstTarget.rel_abs_error;
      const secondError = secondTarget.rel_abs_error;
      const improvement = firstError - secondError; // Positive = second is better, negative = first is better
      const relativeImprovement = firstError > 0 ? improvement / firstError : 0;
      
      let category: 'improved_significantly' | 'worsened' | 'minimal_change';
      
      if (relativeImprovement > 0.2) { // Second dataset improved by more than 20%
        category = 'improved_significantly';
      } else if (relativeImprovement < -0.2) { // Second dataset worsened by more than 20%
        category = 'worsened';
      } else {
        category = 'minimal_change';
      }
      
      targetComparisons.push({
        target_name: firstTarget.target_name,
        firstFinalError: firstError,
        secondFinalError: secondError,
        improvement,
        category
      });
    }
  });

  if (targetComparisons.length === 0) {
    return (
      <div className="bg-white border border-gray-300 p-6 rounded-lg shadow-sm">
        <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <Scale className="w-5 h-5 mr-2" />
          Calibration quality comparison
        </h2>
        <div className="bg-yellow-50 border border-yellow-200 rounded p-4">
          <p className="text-yellow-800">
            No overlapping targets found between datasets for quality comparison.
          </p>
        </div>
      </div>
    );
  }

  // Calculate statistics
  const totalTargets = targetComparisons.length;
  const improvedSignificantly = targetComparisons.filter(t => t.category === 'improved_significantly').length;
  const worsened = targetComparisons.filter(t => t.category === 'worsened').length;
  const minimalChange = targetComparisons.filter(t => t.category === 'minimal_change').length;

  const improvedPercentage = totalTargets > 0 ? (improvedSignificantly / totalTargets * 100) : 0;
  const worsenedPercentage = totalTargets > 0 ? (worsened / totalTargets * 100) : 0;
  const minimalPercentage = totalTargets > 0 ? (minimalChange / totalTargets * 100) : 0;

  // Calculate overall quality assessment
  const averageImprovement = targetComparisons.reduce((sum, t) => sum + t.improvement, 0) / totalTargets;
  const averageFirstError = targetComparisons.reduce((sum, t) => sum + t.firstFinalError, 0) / totalTargets;
  const averageSecondError = targetComparisons.reduce((sum, t) => sum + t.secondFinalError, 0) / totalTargets;
  
  const overallQualityChange = averageImprovement / averageFirstError;
  
  let qualityAssessment: string;
  let qualityColor: string;
  let qualityIcon: React.ReactNode;
  
  if (overallQualityChange > 0.1) {
    qualityAssessment = "Improved";
    qualityColor = "green";
    qualityIcon = <TrendingUp className="w-6 h-6" />;
  } else if (overallQualityChange < -0.1) {
    qualityAssessment = "Worsened";
    qualityColor = "red";
    qualityIcon = <TrendingDown className="w-6 h-6" />;
  } else {
    qualityAssessment = "Similar Quality";
    qualityColor = "gray";
    qualityIcon = <Minus className="w-6 h-6" />;
  }


  return (
    <div className="bg-white border border-gray-300 p-6 rounded-lg shadow-sm">
      <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
        <Scale className="w-5 h-5 mr-2" />
        Calibration quality comparison
      </h2>
      <p className="text-gray-600 mb-6">
        Comparison of final epoch calibration quality between {firstName} and {secondName} across {totalTargets} overlapping targets
      </p>

      {/* Overall Quality Assessment */}
      <div className={`mb-6 border-2 border-${qualityColor}-500 rounded-lg p-6`}>
        <div className="text-center">
          <div className={`flex items-center justify-center mb-2 text-${qualityColor}-600`}>
            {qualityIcon}
            <h3 className={`text-2xl font-bold text-${qualityColor}-800 ml-2`}>
              {qualityAssessment}
            </h3>
          </div>
          <p className={`text-${qualityColor}-700 text-lg mb-2`}>
            {secondName} is {qualityAssessment.toLowerCase()} than {firstName}
          </p>
          <div className={`text-${qualityColor}-600 text-sm`}>
            Average error: {(averageFirstError * 100).toFixed(2)}% → {(averageSecondError * 100).toFixed(2)}% 
            ({overallQualityChange >= 0 ? '-' : '+'}{Math.abs(overallQualityChange * 100).toFixed(1)}% change)
          </div>
        </div>
      </div>

      {/* Summary Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {/* Improved in Second */}
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center mb-2">
            <TrendingUp className="w-5 h-5 text-green-600 mr-2" />
            <h3 className="text-lg font-semibold text-green-800">Better in {secondName}</h3>
          </div>
          <div className="text-2xl font-bold text-green-700">{improvedSignificantly}</div>
          <div className="text-sm text-green-600">
            {improvedPercentage.toFixed(1)}% of targets ({improvedSignificantly}/{totalTargets})
          </div>
          <div className="text-xs text-green-500 mt-1">
            Error reduced by &gt;20%
          </div>
        </div>

        {/* Worse in Second */}
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center mb-2">
            <TrendingDown className="w-5 h-5 text-red-600 mr-2" />
            <h3 className="text-lg font-semibold text-red-800">Better in {firstName}</h3>
          </div>
          <div className="text-2xl font-bold text-red-700">{worsened}</div>
          <div className="text-sm text-red-600">
            {worsenedPercentage.toFixed(1)}% of targets ({worsened}/{totalTargets})
          </div>
          <div className="text-xs text-red-500 mt-1">
            Error increased by &gt;20%
          </div>
        </div>

        {/* Similar Quality */}
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
          <div className="flex items-center mb-2">
            <Minus className="w-5 h-5 text-gray-600 mr-2" />
            <h3 className="text-lg font-semibold text-gray-800">Similar quality</h3>
          </div>
          <div className="text-2xl font-bold text-gray-700">{minimalChange}</div>
          <div className="text-sm text-gray-600">
            {minimalPercentage.toFixed(1)}% of targets ({minimalChange}/{totalTargets})
          </div>
          <div className="text-xs text-gray-500 mt-1">
            Error change within ±20%
          </div>
        </div>
      </div>
    </div>
  );
}
