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
        console.log(day)

        // get the current availability for the day
        let currentTimes = this.userService.eventAvailabilityForm.value[day];
        console.log(currentTimes)

        // remove the unavailable time
        currentTimes = currentTimes.filter((time: number) => time !== 0);
        console.log(currentTimes)

        // add these times to the current availability
        times.forEach((time: number) => {
          if (!currentTimes.includes(time)) {
            currentTimes.push(time);
          }
        });

        // update the availability for the day
        this.userService.eventAvailabilityForm.controls[day].setValue(currentTimes);
        console.log(currentTimes)
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
   * Updates the times for the weekdays.
   */
  updateWeekdayTimes() {
    console.log("Updating weekday times...")
    console.log(this.generalAvailabilityForm.value)

    let weekdayTimes: string[] = ['mondayTimes', 'tuesdayTimes', 'wednesdayTimes', 'thursdayTimes', 'fridayTimes'];
    
    // early morning
    if (this.generalAvailabilityForm.value.weekdayEarlyMorning) {
      this.addTimeRange('Early Morning', weekdayTimes);
      
    } else {
      this.removeTimeRange('Early Morning', weekdayTimes);
    }

    // morning
    if (this.generalAvailabilityForm.value.weekdayMorning) {
      this.addTimeRange('Morning', weekdayTimes);
    } else {
      this.removeTimeRange('Morning', weekdayTimes);
    }

    // afternoon
    if (this.generalAvailabilityForm.value.weekdayAfternoon) {
      this.addTimeRange('Afternoon', weekdayTimes);
    } else {
      this.removeTimeRange('Afternoon', weekdayTimes);
    }

    // evening
    if (this.generalAvailabilityForm.value.weekdayEvening) {
      this.addTimeRange('Evening', weekdayTimes);
    } else {
      this.removeTimeRange('Evening', weekdayTimes);
    }

    // night
    if (this.generalAvailabilityForm.value.weekdayNight) {
      this.addTimeRange('Night', weekdayTimes);
    } else {
      this.removeTimeRange('Night', weekdayTimes);
    }

    // late night
    if (this.generalAvailabilityForm.value.weekdayLateNight) {
      this.addTimeRange('Late Night', weekdayTimes);
    } else {
      this.removeTimeRange('Late Night', weekdayTimes);
    }

    // unavailable
    if (this.generalAvailabilityForm.value.weekdayUnavailable) {
      for (let day of weekdayTimes) {
        this.userService.eventAvailabilityForm.controls[day].setValue([0]);
      }
    }

    console.log(this.userService.eventAvailabilityForm.value);
  }

  /**
   * Updates the times for the weekends.
   */
  updateWeekendTimes() {
    // early morning
    if (this.generalAvailabilityForm.value.weekendEarlyMorning) {
      this.addTimeRange('Early Morning', ['saturdayTimes', 'sundayTimes']);
    } else {
      this.removeTimeRange('Early Morning', ['saturdayTimes', 'sundayTimes']);
    }

    // morning
    if (this.generalAvailabilityForm.value.weekendMorning) {
      this.addTimeRange('Morning', ['saturdayTimes', 'sundayTimes']);
    } else {
      this.removeTimeRange('Morning', ['saturdayTimes', 'sundayTimes']);
    }

    // afternoon
    if (this.generalAvailabilityForm.value.weekendAfternoon) {
      this.addTimeRange('Afternoon', ['saturdayTimes', 'sundayTimes']);
    } else {
      this.removeTimeRange('Afternoon', ['saturdayTimes', 'sundayTimes']);
    }

    // evening
    if (this.generalAvailabilityForm.value.weekendEvening) {
      this.addTimeRange('Evening', ['saturdayTimes', 'sundayTimes']);
    } else {
      this.removeTimeRange('Evening', ['saturdayTimes', 'sundayTimes']);
    }

    // night
    if (this.generalAvailabilityForm.value.weekendNight) {
      this.addTimeRange('Night', ['saturdayTimes', 'sundayTimes']);
    } else {
      this.removeTimeRange('Night', ['saturdayTimes', 'sundayTimes']);
    }

    // late night
    if (this.generalAvailabilityForm.value.weekendLateNight) {
      this.addTimeRange('Late Night', ['saturdayTimes', 'sundayTimes']);
    } else {
      this.removeTimeRange('Late Night', ['saturdayTimes', 'sundayTimes']);
    }

    // unavailable
    if (this.generalAvailabilityForm.value.weekendUnavailable) {
      this.userService.eventAvailabilityForm.controls['saturdayTimes'].setValue([0]);
      this.userService.eventAvailabilityForm.controls['sundayTimes'].setValue([0]);
    }
  }
}
