import { Component } from '@angular/core';
import { FormBuilder, FormGroup, FormControl } from '@angular/forms';

// services
import { UserService } from '../_services/user.service';

// helpers
import { availableTimes, days, timeCategoryMap } from '../_helpers/preferences';

@Component({
  selector: 'app-event-availability-survey',
  templateUrl: './event-availability-survey.component.html',
  styleUrl: './event-availability-survey.component.css'
})
export class EventAvailabilitySurveyComponent {
  // general availability form
  generalAvailabilityForm: FormGroup = this.fb.group({
    weekdayEarlyMorning: new FormControl(false),
    weekdayMorning: new FormControl(false),
    weekdayAfternoon: new FormControl(false),
    weekdayEvening: new FormControl(false),
    weekdayNight: new FormControl(false),
    weekdayLateNight: new FormControl(false),
    weekdayUnavailable: new FormControl(false),
    weekendEarlyMorning: new FormControl(false),
    weekendMorning: new FormControl(false),
    weekendAfternoon: new FormControl(false),
    weekendEvening: new FormControl(false),
    weekendNight: new FormControl(false),
    weekendLateNight: new FormControl(false),
    weekendUnavailable: new FormControl(false),
  });

  weekdayGeneralAvailabilityControls = [
    'weekdayEarlyMorning',
    'weekdayMorning',
    'weekdayAfternoon',
    'weekdayEvening',
    'weekdayNight',
    'weekdayLateNight',
    'weekdayUnavailable'
  ];

  weekendGeneralAvailabilityControls = [
    'weekendEarlyMorning',
    'weekendMorning',
    'weekendAfternoon',
    'weekendEvening',
    'weekendNight',
    'weekendLateNight',
    'weekendUnavailable'
  ];

  weekdayFormControlLabelMapList = [
    {control: 'weekdayUnavailable', label: 'Unavailable'},
    {control: 'weekdayEarlyMorning', label: 'Early Morning (5-8a)'},
    {control: 'weekdayMorning', label: 'Morning (9a-12p)'},
    {control: 'weekdayAfternoon', label: 'Afternoon (1-4p)'},
    {control: 'weekdayEvening', label: 'Evening (5-8p)'},
    {control: 'weekdayNight', label: 'Night (9p-12a)'},
    {control: 'weekdayLateNight', label: 'Late Night (1-4a)'}
  ];

  weekendFormControlLabelMapList = [
    {control: 'weekendUnavailable', label: 'Unavailable'},
    {control: 'weekendEarlyMorning', label: 'Early Morning (5-8a)'},
    {control: 'weekendMorning', label: 'Morning (9a-12p)'},
    {control: 'weekendAfternoon', label: 'Afternoon (1-4p)'},
    {control: 'weekendEvening', label: 'Evening (5-8p)'},
    {control: 'weekendNight', label: 'Night (9p-12a)'},
    {control: 'weekendLateNight', label: 'Late Night (1-4a)'}
  ];

  availableTimes = availableTimes;
  timeCategoryMap = timeCategoryMap;
  days = days;
  constructor(
    public userService: UserService,
    private fb: FormBuilder
  ) {}

  /**
   * Submits the event availability form.
   */
  onSubmit() {
    console.log(this.userService.eventAvailabilityForm.value);
  }

  /**
   * Updates the availability based on the form values.
   */
  updateAvailability() {
    this.updateWeekdayTimes();
    // this.updateWeekendTimes();
  }

  /**
   * Adds the time range to the availability.
   * 
   * @param timeRange the time range to add
   * @param days the days to add the time range to
   */
  addTimeRange(timeRange: string, days: string[]) {
    let times = this.timeCategoryMap[timeRange];
      for (let day of days) {
        // get the current availability for the day
        let currentTimes = this.userService.eventAvailabilityForm.value[day];

        // remove the unavailable time
        currentTimes = currentTimes.filter((time: number) => time !== 0);

        // add these times to the current availability
        times.forEach((time: number) => {
          if (!currentTimes.includes(time)) {
            currentTimes.push(time);
          }
        });

        // update the availability for the day
        this.userService.eventAvailabilityForm.controls[day].setValue(currentTimes);
      } 
  }

  /**
   * Removes the time range from the availability.
   * 
   * @param timeRange the time range to remove
   * @param days the days to remove the time range from
   * @returns the updated availability
   */
  removeTimeRange(timeRange: string, days: string[]) {
    let times = this.timeCategoryMap[timeRange];
    let currentTimes: number[] = [];
    for (let day of days) {
      // get the current availability for the day
      let currentTimes = this.userService.eventAvailabilityForm.value[day];

      // remove the times from the current availability
      times.forEach((time: number) => {
        currentTimes = currentTimes.filter((t: number) => t !== time);
      });

      // update the form control
      this.userService.eventAvailabilityForm.controls[day].setValue(currentTimes);
    }
    return currentTimes;
  }

  /**
   * Get the disabled state for the weekday controls.
   * 
   * @param control The control to get the disabled state of.
   * @returns If the control should be disabled.
   */
  getWeekdayDisabledState(control: string) {
    if (control.toLowerCase().includes('unavailable')) {
      for (let weekdayControl of this.weekdayGeneralAvailabilityControls) {
        if (this.generalAvailabilityForm.get(weekdayControl)?.value && weekdayControl !== control) {
          return true;
        }
      }
    } else {
      return this.generalAvailabilityForm.get('weekdayUnavailable')?.value
    }
  }

  /**
   * Get the disabled state for the weekend controls.
   * 
   * @param control The control to get the disabled state of.
   * @returns If the control should be disabled.
   */
  getWeekendDisabledState(control: string) {
    if (control.toLowerCase().includes('unavailable')) {
      for (let weekendControl of this.weekendGeneralAvailabilityControls) {
        if (this.generalAvailabilityForm.get(weekendControl)?.value && weekendControl !== control) {
          return true;
        }
      }
    } else {
      return this.generalAvailabilityForm.get('weekendUnavailable')?.value
    }
  }

  /**
   * Updates the times for the weekdays.
   */
  updateWeekdayTimes() {
    let weekdayTimes: string[] = ['mondayTimes', 'tuesdayTimes', 'wednesdayTimes', 'thursdayTimes', 'fridayTimes'];
    
    // early morning
    if (this.generalAvailabilityForm.value.weekdayEarlyMorning) {
      this.addTimeRange('Early morning (5-8a)', weekdayTimes);
      
    } else {
      this.removeTimeRange('Early morning (5-8a)', weekdayTimes);
    }

    // morning
    if (this.generalAvailabilityForm.value.weekdayMorning) {
      this.addTimeRange('Morning (9a-12p)', weekdayTimes);
    } else {
      this.removeTimeRange('Morning (9a-12p)', weekdayTimes);
    }

    // afternoon
    if (this.generalAvailabilityForm.value.weekdayAfternoon) {
      this.addTimeRange('Afternoon (1-4p)', weekdayTimes);
    } else {
      this.removeTimeRange('Afternoon (1-4p)', weekdayTimes);
    }

    // evening
    if (this.generalAvailabilityForm.value.weekdayEvening) {
      this.addTimeRange('Evening (5-8p)', weekdayTimes);
    } else {
      this.removeTimeRange('Evening (5-8p)', weekdayTimes);
    }

    // night
    if (this.generalAvailabilityForm.value.weekdayNight) {
      this.addTimeRange('Night (9p-12a)', weekdayTimes);
    } else {
      this.removeTimeRange('Night (9p-12a)', weekdayTimes);
    }

    // late night
    if (this.generalAvailabilityForm.value.weekdayLateNight) {
      this.addTimeRange('Late night (1-4a)', weekdayTimes);
    } else {
      this.removeTimeRange('Late night (1-4a)', weekdayTimes);
    }

    // unavailable
    if (this.generalAvailabilityForm.value.weekdayUnavailable) {
      for (let day of weekdayTimes) {
        this.userService.eventAvailabilityForm.controls[day].setValue([0]);
      }
    } else {
      this.removeTimeRange('Unavailable', weekdayTimes);
    }
  }

  /**
   * Updates the times for the weekends.
   */
  updateWeekendTimes() {
    // early morning
    if (this.generalAvailabilityForm.value.weekendEarlyMorning) {
      this.addTimeRange('Early morning (5-8a)', ['saturdayTimes', 'sundayTimes']);
    } else {
      this.removeTimeRange('Early morning (5-8a)', ['saturdayTimes', 'sundayTimes']);
    }

    // morning
    if (this.generalAvailabilityForm.value.weekendMorning) {
      this.addTimeRange('Morning (9a-12p)', ['saturdayTimes', 'sundayTimes']);
    } else {
      this.removeTimeRange('Morning (9a-12p)', ['saturdayTimes', 'sundayTimes']);
    }

    // afternoon
    if (this.generalAvailabilityForm.value.weekendAfternoon) {
      this.addTimeRange('Afternoon (1-4p)', ['saturdayTimes', 'sundayTimes']);
    } else {
      this.removeTimeRange('Afternoon (1-4p)', ['saturdayTimes', 'sundayTimes']);
    }

    // evening
    if (this.generalAvailabilityForm.value.weekendEvening) {
      this.addTimeRange('Evening (5-8p)', ['saturdayTimes', 'sundayTimes']);
    } else {
      this.removeTimeRange('Evening (5-8p)', ['saturdayTimes', 'sundayTimes']);
    }

    // night
    if (this.generalAvailabilityForm.value.weekendNight) {
      this.addTimeRange('Night (9p-12a)', ['saturdayTimes', 'sundayTimes']);
    } else {
      this.removeTimeRange('Night (9p-12a)', ['saturdayTimes', 'sundayTimes']);
    }

    // late night
    if (this.generalAvailabilityForm.value.weekendLateNight) {
      this.addTimeRange('Late night (1-4a)', ['saturdayTimes', 'sundayTimes']);
    } else {
      this.removeTimeRange('Late night (1-4a)', ['saturdayTimes', 'sundayTimes']);
    }

    // unavailable
    if (this.generalAvailabilityForm.value.weekendUnavailable) {
      this.userService.eventAvailabilityForm.controls['saturdayTimes'].setValue([0]);
      this.userService.eventAvailabilityForm.controls['sundayTimes'].setValue([0]);
    } else {
      this.removeTimeRange('Unavailable', ['saturdayTimes', 'sundayTimes']);
    }
  }
}
